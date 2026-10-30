import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from data import shared_splits, loader, validate_splits
from models import make_model, set_level, Backbone
from structure import Structure
from methods import WeightedAccumulator, local_train, estimate_importance, proximal_penalty, distillation, METHODS
from evaluation_core import classification_metrics
from run import check_compaction, seed_all

torch.set_num_threads(2)
CFG = json.loads((Path(__file__).parent / 'configs/fashionmnist.json').read_text())


class CorrectnessTests(unittest.TestCase):
    def test_compact_equivalence_all_backbones_and_exits(self):
        for dataset, channels, size in [('fashionmnist', 1, 28), ('cifar10', 3, 32), ('cifar100', 3, 32)]:
            with self.subTest(dataset=dataset):
                seed_all(42)
                model = make_model(dataset, exits=True).eval()
                # Nontrivial BN affine and running buffers expose incorrect zero masking.
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.running_mean.fill_(.13)
                        module.running_var.fill_(1.3)
                        module.bias.data.fill_(.17)
                s = Structure(model)
                x = torch.randn(2, channels, size, size)
                before = model(x).detach()
                for p in (0, 20, 40):
                    mask = s.masks(p, kind='random', generator=torch.Generator().manual_seed(42))
                    compact = s.compact(model, mask)
                    for exit in list(range(len(model.exits))) + ['final']:
                        check_compaction(model, compact, s, mask, x, p, exit)
                    self.assertTrue(torch.equal(before, model(x).detach()))

    def test_switchable_bn_banks_and_compaction(self):
        model = make_model('fashionmnist', switch_bn=True).train()
        s = Structure(model)
        x = torch.randn(3, 1, 28, 28)
        bn = model.stages[0][1]
        full = bn.running_mean.clone()
        set_level(model, 40)
        model(x)
        self.assertTrue(torch.equal(full, bn.running_mean))
        self.assertFalse(torch.equal(full, bn.banks['40'].running_mean))
        mask = s.masks(40, kind='prefix')
        compact = s.compact(model, mask, 40)
        check_compaction(model, compact, s, mask, x, 40, 'final')

    def test_nested_exact_budget_and_infeasibility(self):
        s = Structure(make_model('cifar100'))
        scores = s.zeros()
        previous = s.masks(0, scores)
        for p in (5, 10, 15, 20, 25, 30, 40):
            masks = s.masks(p, scores)
            self.assertEqual(sum(int((~masks[g]).sum()) for g in s.eligible), round(p * s.total / 100))
            for g in masks:
                self.assertTrue(torch.all(masks[g] <= previous[g]))
                self.assertLessEqual(int((~masks[g]).sum()), s.caps[g])
            previous = masks
        with self.assertRaises(ValueError):
            s.masks(99, scores)

    def test_mask_restores_weights_and_masks_task_gradients(self):
        model = make_model('fashionmnist')
        s = Structure(model)
        mask = s.masks(40, s.zeros())
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        with s.applied(model, mask, mask_gradients=True):
            model(torch.randn(2, 1, 28, 28)).sum().backward()
        for n, p in model.named_parameters():
            self.assertTrue(torch.equal(p, before[n]))
        for n, m in s.parameter_masks(model, mask).items():
            grad = dict(model.named_parameters())[n].grad
            if grad is not None:
                self.assertTrue(torch.all(grad[~m.expand_as(grad)] == 0))

    def test_importance_does_not_mutate_checkpoint(self):
        model = make_model('fashionmnist')
        before = copy.deepcopy(model.state_dict())
        ds = TensorDataset(torch.randn(4, 1, 28, 28), torch.tensor([1, 2, 3, 4]))
        s = Structure(model)
        scores = estimate_importance(model, s, loader(ds, list(range(4)), 2, 42), 2, torch.device('cpu'))
        self.assertTrue(all(torch.equal(before[n], v) for n, v in model.state_dict().items()))
        self.assertGreater(sum(float(v.sum()) for v in scores.values()), 0)

    def test_importance_is_sum_of_absolute_products(self):
        model = Backbone([nn.Sequential(nn.Conv2d(1, 3, 1, bias=False), nn.ReLU())],
                         [3], nn.Linear(3, 2), 2)
        ds = TensorDataset(torch.randn(4, 1, 3, 3), torch.tensor([0, 1, 0, 1]))
        s = Structure(model)
        score = estimate_importance(model, s, loader(ds, list(range(4)), 4, 42), 1, torch.device('cpu'))
        torch.nn.functional.cross_entropy(model(ds.tensors[0]), ds.tensors[1]).backward()
        weight = model.stages[0][0].weight
        expected = (weight * weight.grad).abs().flatten(1).sum(1)
        torch.testing.assert_close(score['stages.0.0'], expected.detach())

    def test_common_backbone_initialization(self):
        seed_all(42)
        common = make_model('fashionmnist').state_dict()
        for method in METHODS:
            seed_all(42)
            state = make_model('fashionmnist', exits=method == 'slex_style', switch_bn=method == 'slimmable').state_dict()
            self.assertTrue(all(torch.equal(value, state[key]) for key, value in common.items()))

    def test_residual_and_depthwise_training(self):
        for dataset, size, classes in [('cifar10', 32, 10), ('cifar100', 32, 100)]:
            for method in ('rasfl', 'slex_style'):
                with self.subTest(dataset=dataset, method=method):
                    model = make_model(dataset, exits=method == 'slex_style')
                    s = Structure(model)
                    ds = TensorDataset(torch.randn(2, 3, size, size), torch.tensor([0, classes - 1]))
                    local_train(model, s, loader(ds, [0, 1], 2, 42, True), method,
                                dict(CFG, warmup_rounds=0, mask_probability=1.), 51, s.zeros(), 42, torch.device('cpu'))
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_proximal_and_distillation_formulas(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(3)
        reference = {'weight': torch.ones_like(model.weight)}
        self.assertAlmostEqual(float(proximal_penalty(model, reference).detach()), 4.)
        z = torch.tensor([[1., 2., 3.]])
        self.assertAlmostEqual(float(distillation(z, z, 2)), 0., places=6)

    def test_weighted_aggregation(self):
        a, b = nn.BatchNorm2d(2), nn.BatchNorm2d(2)
        a.weight.data.fill_(1)
        b.weight.data.fill_(5)
        a.num_batches_tracked.fill_(2)
        b.num_batches_tracked.fill_(3)
        avg = WeightedAccumulator()
        avg.add(a, 1)
        avg.add(b, 3)
        state = avg.finish()
        self.assertTrue(torch.equal(state['weight'], torch.full((2,), 4.)))
        self.assertEqual(int(state['num_batches_tracked']), 3)

    def test_disjoint_saved_splits(self):
        labels = (np.arange(4000) % 10).tolist()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'split.json'
            a = shared_splits(CFG, 42, labels, labels, {'0': 0}, path)
            b = shared_splits(CFG, 42, labels, labels, {'0': 0}, path)
            self.assertEqual(a, b)
            self.assertEqual(len(a['train_clients']), 20)
            self.assertEqual(len(a['deployment_clients']), 20)
            validate_splits(a, len(labels), len(labels))
            with self.assertRaises(ValueError):
                shared_splits(CFG, 123, labels, labels, {'0': 0}, path)

    def test_no_accuracy_f1_fallback(self):
        m = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0], 3)
        self.assertAlmostEqual(m['acc'], .75)
        self.assertAlmostEqual(m['macro_f1'], (6 / 7) / 3)
        self.assertAlmostEqual(m['balanced_acc'], .5)

    def test_all_local_objectives_run_and_change_weights(self):
        ds = TensorDataset(torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 3]))
        cfg = dict(CFG, warmup_rounds=0, mask_probability=1.)
        for method in METHODS:
            with self.subTest(method=method):
                seed_all(42)
                model = make_model('fashionmnist', exits=method == 'slex_style', switch_bn=method == 'slimmable')
                initial = model.stages[0][0].weight.detach().clone()
                s = Structure(model)
                cost = local_train(model, s, loader(ds, list(range(4)), 4, 42, True), method, cfg, 1, s.zeros(), 42, torch.device('cpu'))
                self.assertFalse(torch.equal(initial, model.stages[0][0].weight))
                self.assertGreater(cost['optimizer_steps'], 0)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_fedprox_zero_mu_matches_fedavg(self):
        ds = TensorDataset(torch.randn(6, 1, 28, 28), torch.arange(6))
        base = make_model('fashionmnist')
        result = []
        for method in ('fedavg', 'fedprox'):
            seed_all(42)
            model = copy.deepcopy(base)
            s = Structure(model)
            local_train(model, s, loader(ds, list(range(6)), 3, 42, True), method, dict(CFG, prox_mu=0), 1, s.zeros(), 42, torch.device('cpu'))
            result.append(model.state_dict())
        self.assertTrue(all(torch.equal(result[0][k], result[1][k]) for k in result[0]))

    def test_singleton_tail_keeps_every_sample(self):
        ds = TensorDataset(torch.arange(9), torch.arange(9))
        batches = list(loader(ds, list(range(9)), 4, 42, True))
        self.assertEqual([len(y) for _, y in batches], [4, 5])
        self.assertEqual(sorted(torch.cat([y for _, y in batches]).tolist()), list(range(9)))


if __name__ == '__main__':
    unittest.main()
