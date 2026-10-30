import copy
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from algorithms import FIARSE, FjORD, ThresholdWeight, PartialAverage, configure, parameter_targets, local_train
from models import make_model, LEVELS

class AlgorithmsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def check_threshold(self, device):
        old = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            model = FIARSE().to(device)
            params = dict(model.base.named_parameters())
            values = torch.cat([p.detach().abs().flatten().cpu() for n,p in params.items() if n in model.names])
            dense = sum(p.numel() for n,p in params.items() if n not in model.names)
            for target in parameter_targets().values():
                k = max(1, min(values.numel(), target-dense))
                expected = torch.tensor(0.) if k == values.numel() else values.sort().values[-k]
                count = model.set_budget(target)
                torch.testing.assert_close(model.threshold.cpu(), expected, rtol=0, atol=0)
                self.assertEqual(model.threshold.device, next(model.parameters()).device)
                self.assertEqual(count, dense + int((values >= expected).sum()))
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(old, warn_only=old_warn)

    def test_deterministic_threshold_cpu(self):
        self.check_threshold('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_deterministic_threshold_cuda(self):
        self.check_threshold('cuda')

    def test_tcb_gradient(self):
        w = torch.tensor([-2., -.5, 0., .5, 2.], requires_grad=True)
        ThresholdWeight.apply(w, torch.tensor(1.)).sum().backward()
        torch.testing.assert_close(w.grad, torch.tensor([13/9,0.,0.,0.,13/9]))

    def test_full_forward_equivalence(self):
        base = make_model('fashionmnist').eval()
        x = torch.randn(4,1,28,28)
        for cls in (FIARSE,FjORD):
            model = cls(copy.deepcopy(base)).eval()
            if cls == FIARSE:
                model.set_budget(sum(p.numel() for p in base.parameters()))
            torch.testing.assert_close(model(x),base(x))

    def test_nested_support_and_counts(self):
        m = FjORD()
        previous = m.support(0)
        counts = parameter_targets()
        self.assertEqual(counts[0],sum(p.numel() for p in m.base.parameters()))
        for removal in LEVELS[1:]:
            current=m.support(removal)
            self.assertTrue(all(not bool((current[k] & ~previous[k]).any()) for k in current))
            self.assertEqual(sum(int(v.sum()) for v in current.values()),counts[removal])
            previous=current

    def test_partial_averaging_no_dilution(self):
        m = torch.nn.Linear(2,1,bias=False)
        with torch.no_grad(): m.weight.fill_(10)
        a = PartialAverage(m)
        with torch.no_grad(): m.weight.copy_(torch.tensor([[2.,99.]]))
        a.add(m,1,{'weight':torch.tensor([[True,False]])})
        with torch.no_grad(): m.weight.copy_(torch.tensor([[4.,8.]]))
        a.add(m,3,{'weight':torch.tensor([[True,True]])})
        torch.testing.assert_close(a.finish()['weight'],torch.tensor([[3.5,8.]]))
        untouched=PartialAverage(m)
        torch.testing.assert_close(untouched.finish()['weight'],m.weight)

    def test_local_masks_and_bn_banks(self):
        dl=DataLoader(TensorDataset(torch.randn(8,1,28,28),torch.arange(8)),batch_size=4)
        cfg={'lr':.01,'momentum':.9,'weight_decay':.0005,'local_epochs':1}
        for name,cls in [('fiarse',FIARSE),('fjord',FjORD)]:
            m=cls()
            configure(m,40,parameter_targets())
            before={n:p.detach().clone() for n,p in m.named_parameters()}
            support=m.support()
            used=local_train(m,dl,name,40,parameter_targets(),cfg,42,'cpu')
            for n,p in m.named_parameters():
                torch.testing.assert_close(p.detach()[~support[n]],before[n][~support[n]])
            self.assertTrue(any(bool(v.any()) for v in used.values()))
            if name=='fjord':
                self.assertEqual(int(m.banks['s0_n20'].num_batches_tracked),2)
                self.assertEqual(int(m.banks['s0_n32'].num_batches_tracked),0)

if __name__=='__main__': unittest.main()
