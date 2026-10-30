"""Train, select, and evaluate paper-aligned 20-client experiments.

Example: python run.py --config configs/cifar10.json --method all --download
Use --smoke for synthetic plumbing checks, never for reported accuracies.
"""
import argparse
import copy
import csv
import hashlib
import gzip
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
# Required by deterministic CUDA matrix multiplication; set before CUDA initialization.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import TensorDataset
import torchvision
from data import load_data, loader, shared_splits, digest
from models import make_model, LEVELS, set_level
from structure import Structure
from methods import METHODS, WeightedAccumulator, estimate_importance, ema, local_train
from evaluation_core import classification_metrics
from early_stopping import EarlyStopping


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str((value.dtype, list(value.shape))).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def metric_pass(model, data_loader, classes, device, exit_index='final', return_predictions=False):
    targets, predictions = [], []
    loss_sum = 0.
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            logits = model.forward_exit(x.to(device), exit_index)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, y.to(device), reduction='sum'))
            targets.extend(y.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    result = classification_metrics(targets, predictions, classes)
    result['loss'] = loss_sum / len(targets)
    if return_predictions:
        result['_targets'], result['_predictions'] = targets, predictions
    return result


def validation_score(model, structure, masks, loaders, classes, device, adaptive,
                     full_weight=.5, details=None, levels=None):
    values = {}
    for p in (levels if levels is not None else ([0, 10, 20, 30, 40] if adaptive else [0])):
        set_level(model, p)
        with structure.applied(model, masks[p]):
            per_client = [metric_pass(model, dl, classes, device) for dl in loaders]
            values[p] = float(np.mean([m['acc'] for m in per_client]))
            if details is not None:
                details[p] = {k: float(np.mean([m[k] for m in per_client])) for k in ('acc', 'macro_f1', 'balanced_acc', 'loss')}
    score = full_weight * values[0] + (1 - full_weight) * np.mean([values[p] for p in values if p]) if adaptive else values[0]
    set_level(model, 0)
    return float(score), values


def roadmap(structure, method, scores, random_scores, ordering='importance'):
    if method == 'rasfl':
        if ordering == 'prefix':
            return {p: structure.masks(p, kind='prefix') for p in LEVELS}
        if ordering == 'random':
            scores = random_scores
        return {p: structure.masks(p, scores) for p in LEVELS}
    if method == 'feddrop_style':
        return {p: structure.masks(p, random_scores) for p in LEVELS}
    return {p: structure.masks(p, kind='prefix') for p in LEVELS}


def check_compaction(model, compact, structure, masks, x, level, exit_index):
    model.eval()
    set_level(model, level)
    compact.eval()
    # Compare both shapes in full float32 precision. Restore the caller's
    # settings even if inference or the equivalence assertion fails.
    old_conv = torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad(), structure.applied(model, masks):
            dense = model.forward_exit(x, exit_index)
        with torch.no_grad():
            small = compact.forward_exit(x, exit_index)
        torch.testing.assert_close(dense, small, rtol=1e-4, atol=1e-5)
        return float((dense - small).abs().max())
    finally:
        torch.backends.cudnn.allow_tf32 = old_conv
        torch.backends.cuda.matmul.allow_tf32 = old_matmul


def used_parameters(model, exit_index):
    # Count deployed path only, excluding unused heads and later stages.
    modules = [model.head] + list(model.stages) if exit_index == 'final' else list(model.stages)[:int(exit_index) + 1] + [model.exits[int(exit_index)]]
    return sum(p.numel() for module in modules for p in module.parameters())


def evaluate_deployment(model, structure, method, masks_by_level, scores, cfg,
                        split, heldout, seed, device, destination=None):
    rows = []
    prediction_file = gzip.open(destination / 'predictions.jsonl.gz', 'wt') if destination else None
    saved_roadmaps, adaptation_records = {}, []
    levels = [0] if method in ('fedavg', 'fedprox') else LEVELS
    exits = list(range(len(model.exits))) + ['final'] if method == 'slex_style' else ['final']
    for cid, indices in enumerate(split['deployment_clients']):
        dl = loader(heldout, indices['test'], cfg['batch_size'], seed)
        x = next(iter(dl))[0][:2].to(device)
        variants = [('global', masks_by_level)]
        if method == 'rasfl' and cfg.get('ordering', 'importance') == 'importance':
            seed_all(seed + 900_000 + cid)
            adapt = loader(heldout, indices['adapt'], cfg['batch_size'], seed + cid)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            adaptation_start = time.perf_counter()
            local = estimate_importance(model, structure, adapt, cfg['importance_batches'], device)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            adaptation_seconds = time.perf_counter() - adaptation_start
            # Supplied method: global weight 1/(1 + gamma * actual adaptation count).
            mix = 1 - 1 / (1 + cfg['personalization_gamma'] * len(indices['adapt']))
            blended = {g: (1 - mix) * scores[g] + mix * local[g] for g in scores}
            variants.append(('personalized', {p: structure.masks(p, blended) for p in LEVELS}))
            adaptation_records.append({'client': cid, 'n_adaptation': len(indices['adapt']),
                'local_score_weight': mix, 'importance_seconds': adaptation_seconds,
                'local_scores': {g: v.tolist() for g, v in local.items()}})
        for variant, roadmap_masks in variants:
            if variant == 'personalized':
                saved_roadmaps[str(cid)] = {str(p): {g: m.tolist() for g, m in masks.items()} for p, masks in roadmap_masks.items()}
            for p in levels:
                masks = roadmap_masks[p]
                compact = structure.compact(model, masks, p)
                for exit_index in exits:
                    error = check_compaction(model, compact, structure, masks, x, p, exit_index)
                    metrics = metric_pass(compact, dl, cfg['classes'], device, exit_index, return_predictions=destination is not None)
                    if prediction_file:
                        prediction_file.write(json.dumps({'client': cid, 'variant': variant, 'requested_removal': p,
                            'exit': str(exit_index), 'sample_indices': indices['test'],
                            'targets': metrics.pop('_targets'), 'predictions': metrics.pop('_predictions')}) + '\n')
                    removed = sum(int((~masks[g]).sum()) for g in structure.eligible)
                    active_rows = sum(int(masks[out].sum()) for name, (_, out) in structure.specs.items() if not name.startswith('exits.'))
                    total_rows = sum(structure.sizes[out] for name, (_, out) in structure.specs.items() if not name.startswith('exits.'))
                    rows.append({'method': method, 'variant': variant, 'seed': seed, 'client': cid,
                                 'requested_removal': p, 'achieved_group_removal': 100 * removed / structure.total,
                                 'removed_group_coordinates': removed, 'eligible_group_coordinates': structure.total,
                                 'active_output_structures': active_rows, 'total_output_structures': total_rows,
                                 'active_output_structure_ratio': active_rows / total_rows,
                                 'structure_count_scope': 'entire_backbone_including_protected_final_head_excluding_auxiliary_heads',
                                 'n_adaptation': len(indices['adapt']),
                                 'exit': str(exit_index), 'compact_path_parameters': used_parameters(compact, exit_index),
                                 'compact_logit_max_error': error, **metrics})
                del compact
    set_level(model, 0)
    if prediction_file:
        prediction_file.close()
        with gzip.open(destination / 'personalized_roadmaps.json.gz', 'wt') as handle:
            json.dump(saved_roadmaps, handle)
        with gzip.open(destination / 'adaptation_records.json.gz', 'wt') as handle:
            json.dump(adaptation_records, handle)
    return rows


def run_one(cfg, method, seed, train, clean, heldout, split, destination, device, smoke, resume=False):
    cfg = dict(cfg)
    existed = destination.exists()
    if existed and not resume:
        raise FileExistsError(f'{destination} exists; use --resume or a new output folder')
    destination.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = make_model(cfg['dataset'], exits=method == 'slex_style', switch_bn=method == 'slimmable').to(device)
    structure = Structure(model, cfg['min_active_ratio'], cfg['first_min_active_ratio'])
    identity = {'config': cfg, 'method': method, 'seed': seed, 'synthetic_smoke': smoke,
                'split_sha256': digest(split), 'initial_state_sha256': state_hash(model.state_dict()),
                'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
                'device': str(device), 'cuda_runtime': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                'channel_budget_unit': 'dependency_group_coordinate', 'group_sizes': structure.sizes,
                'group_removal_caps': structure.caps, 'parameter_dependencies': structure.specs,
                'baseline_status': 'paper-inspired implementation with declared defaults; not historical-result reproduction'}
    if existed:
        previous = json.loads((destination / 'manifest.json').read_text())
        for key in ('config', 'method', 'seed', 'synthetic_smoke', 'split_sha256', 'source_sha256'):
            if previous[key] != identity[key]:
                raise ValueError(f'Resume identity changed: {key}; use a new output folder')
        if previous.get('evaluation_complete'):
            print(f'Skip completed {destination}', flush=True)
            return
    write_json(destination / 'manifest.json', identity)
    validation = [loader(clean, c['validation'], cfg['batch_size'], seed) for c in split['train_clients']]
    scores = structure.zeros()
    generator = torch.Generator().manual_seed(seed + 400_000)
    random_scores = {g: torch.rand(n, generator=generator) for g, n in structure.sizes.items() if g in structure.eligible}
    best, history, start_round = -float('inf'), [], 1
    stopper = EarlyStopping(cfg.get('early_stopping_patience', 5), cfg.get('early_stopping_min_delta', 0.))
    if existed and (destination / 'last.pt').exists():
        last = torch.load(destination / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(last['model'])
        scores = {g: v.cpu() for g, v in last['scores'].items()}
        best, history, start_round = last['best'], last['history'], last['round'] + 1
        if 'early_stopping' in last:
            stopper = EarlyStopping(**last['early_stopping'])
        else:
            for record in history:
                stopper.update(record['selection_score'])
    for round_number in range(start_round, start_round if stopper.stopped else cfg['rounds'] + 1):
        round_started = time.perf_counter()
        previous_masks = roadmap(structure, method, scores, random_scores, cfg.get('ordering', 'importance'))
        aggregate, score_sum, total = WeightedAccumulator(), structure.zeros(), 0
        cost = {'student_forward_passes': 0, 'optimizer_steps': 0,
                'objective_loss_sum_over_student_passes': 0., 'objective_student_pass_count': 0}
        for cid, indices in enumerate(split['train_clients']):
            local_seed = seed + round_number * 10_000 + cid
            seed_all(local_seed)
            dl = loader(train, indices['train'], cfg['batch_size'], local_seed, shuffle=True)
            n = len(indices['train'])
            if method == 'rasfl':
                importance_loader = loader(train, indices['train'], cfg['batch_size'], local_seed + 5_000, shuffle=True)
                client_scores = estimate_importance(model, structure, importance_loader, cfg['importance_batches'], device)
                for g in scores:
                    score_sum[g] += n * client_scores[g]
            # Reset augmentation/dropout RNG after importance passes for paired training order.
            seed_all(local_seed)
            local = copy.deepcopy(model)
            training_scores = random_scores if cfg.get('ordering') == 'random' else scores
            measured = local_train(local, structure, dl, method, cfg, round_number, training_scores, local_seed, device)
            aggregate.add(local, n)
            total += n
            for k in cost:
                cost[k] += measured[k]
            del local
        model.load_state_dict(aggregate.finish())
        if method == 'rasfl':
            scores = ema(scores, {g: v / total for g, v in score_sum.items()}, cfg['importance_beta'])
        masks = roadmap(structure, method, scores, random_scores, cfg.get('ordering', 'importance'))
        # All adaptive methods use the same validation capacities and criterion, including warmup.
        validation_metrics = {}
        selection, values = validation_score(model, structure, masks, validation, cfg['classes'], device,
                                              method not in ('fedavg', 'fedprox'),
                                              cfg.get('full_validation_weight', .5), validation_metrics)
        intersection = sum(int((previous_masks[40][g] & masks[40][g]).sum()) for g in structure.eligible)
        union = sum(int((previous_masks[40][g] | masks[40][g]).sum()) for g in structure.eligible)
        history.append({'round': round_number, 'selection_score': selection, 'validation_accuracy': values,
                        'validation_metrics': validation_metrics, 'round_wall_seconds': time.perf_counter() - round_started,
                        'retained_set_jaccard_at_40_previous_round': intersection / union,
                        'mean_objective_over_student_passes': cost['objective_loss_sum_over_student_passes'] / cost['objective_student_pass_count'], **cost})
        stopper.update(selection)
        history[-1]['early_stopping_bad_rounds'] = stopper.bad_rounds
        if selection > best:
            best = selection
            torch.save({'model': model.state_dict(), 'scores': scores, 'round': round_number,
                        'selection_score': best, 'config': cfg, 'method': method, 'seed': seed,
                        'split_sha256': digest(split), 'random_scores': random_scores}, destination / 'best.pt')
        write_json(destination / 'history.json', history)
        torch.save({'model': model.state_dict(), 'scores': scores, 'round': round_number,
                    'best': best, 'history': history, 'early_stopping': stopper.state_dict()}, destination / 'last.tmp.pt')
        (destination / 'last.tmp.pt').replace(destination / 'last.pt')
        print(f'{cfg["dataset"]} {method} seed={seed} round={round_number}/{cfg["rounds"]} val={selection:.4f}', flush=True)
        if stopper.stopped:
            print(f'Early stopping: {stopper.bad_rounds} consecutive rounds without validation improvement. Restoring best checkpoint.', flush=True)
            break
    write_json(destination / 'training_stop.json', {
        'reason': 'early_stopping' if stopper.stopped else 'maximum_rounds',
        'completed_rounds': history[-1]['round'], 'maximum_rounds': cfg['rounds'],
        'patience': stopper.patience, 'min_delta': stopper.min_delta,
        'consecutive_non_improving_rounds': stopper.bad_rounds,
        'monitor': 'validation selection_score; test data never used',
        'monitoring_start': 'round 1, including warmup'})
    selected = torch.load(destination / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['model'])
    scores = {g: v.cpu() for g, v in selected['scores'].items()}
    # Fixed-checkpoint score stabilization only uses training data; no parameter/buffer mutation.
    if method == 'rasfl':
        for step in range(cfg['stabilization_passes']):
            total, combined = 0, structure.zeros()
            for cid, indices in enumerate(split['train_clients']):
                local_seed = seed + 600_000 + step * 10_000 + cid
                seed_all(local_seed)
                dl = loader(train, indices['train'], cfg['batch_size'], local_seed, shuffle=True)
                values = estimate_importance(model, structure, dl, cfg['importance_batches'], device)
                n = len(indices['train'])
                total += n
                for g in combined:
                    combined[g] += n * values[g]
            scores = ema(scores, {g: v / total for g, v in combined.items()}, cfg['importance_beta'])
    masks = roadmap(structure, method, scores, random_scores, cfg.get('ordering', 'importance'))
    final_metrics = {}
    selection, values = validation_score(model, structure, masks, validation, cfg['classes'], device,
                                          method not in ('fedavg', 'fedprox'), cfg.get('full_validation_weight', .5), final_metrics)
    # Additional levels support validation-only dynamic selection, without changing checkpoint choice.
    policy_metrics = {}
    _, policy_values = validation_score(model, structure, masks, validation, cfg['classes'], device,
        method not in ('fedavg', 'fedprox'), cfg.get('full_validation_weight', .5), policy_metrics,
        list(LEVELS) if method not in ('fedavg', 'fedprox') else [0])
    write_json(destination / 'final_validation.json', {'selected_round': selected['round'],
               'selection_before_stabilization': selected['selection_score'],
               'score_after_stabilization': selection, 'accuracy': values, 'metrics': final_metrics,
               'policy_accuracy_all_levels': policy_values, 'policy_metrics_all_levels': policy_metrics})
    torch.save({'model': model.state_dict(), 'scores': scores, 'masks': masks, 'config': cfg,
                'method': method, 'seed': seed, 'selected_round': selected['round'],
                'split_sha256': digest(split)}, destination / 'deployment.pt')
    rows = evaluate_deployment(model, structure, method, masks, scores, cfg, split, heldout, seed, device, destination)
    with (destination / 'clients.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    identity['deployment_checkpoint_sha256'] = hashlib.sha256((destination / 'deployment.pt').read_bytes()).hexdigest()
    identity['evaluation_complete'] = True
    write_json(destination / 'manifest.json', identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--output', default='./runs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume identical runs; skip already completed methods')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = json.loads(Path(args.config).read_text())
    if cfg['training_clients'] != 20:
        raise ValueError('This experiment requires exactly 20 training clients')
    if args.rounds is not None:
        cfg['rounds'] = args.rounds
    if cfg['rounds'] < 1:
        raise ValueError('At least one round is required')
    if args.smoke:
        cfg.update(rounds=1, stabilization_passes=1, importance_batches=1, batch_size=8,
                   warmup_rounds=0)
        gen = torch.Generator().manual_seed(777)
        c, size = (1, 28) if cfg['dataset'] == 'fashionmnist' else (3, 64 if cfg['dataset'] == 'tinyimagenet' else 32)
        train = TensorDataset(torch.randn(240, c, size, size, generator=gen), torch.arange(240) % cfg['classes'])
        heldout = TensorDataset(torch.randn(240, c, size, size, generator=gen), torch.arange(240) % cfg['classes'])
        clean, class_map = train, {str(i): i for i in range(cfg['classes'])}
        train_labels, test_labels = train.tensors[1].tolist(), heldout.tensors[1].tolist()
    else:
        train, clean, heldout, class_map = load_data(cfg, args.data_root, args.download)
        # Fashion-MNIST targets are a Tensor: list(tensor) leaves scalar Tensor
        # objects, whose set membership is not integer-label membership.
        # Normalize both pools before validation, partitioning, and provenance.
        train_labels = [int(label) for label in train.targets]
        test_labels = [int(label) for label in heldout.targets]
        observed_classes = set(train_labels)
        expected_classes = set(range(cfg['classes']))
        if observed_classes != expected_classes:
            raise ValueError('Training class mapping is not the declared complete task: '
                             f'missing={sorted(expected_classes - observed_classes)}, '
                             f'unexpected={sorted(observed_classes - expected_classes)}')
    base = Path(args.output) / ('SMOKE_SYNTHETIC' if args.smoke else 'experiments') / cfg['dataset']
    for seed in args.seeds or cfg['seeds']:
        split = shared_splits(cfg, seed, train_labels, test_labels, class_map, base / f'seed_{seed}' / 'splits.json')
        for method in METHODS if args.method == 'all' else [args.method]:
            run_one(cfg, method, seed, train, clean, heldout, split, base / f'seed_{seed}' / method, torch.device(args.device), args.smoke, args.resume)


if __name__ == '__main__':
    main()
