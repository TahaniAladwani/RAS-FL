"""Execute a declared common memory-budget trace; no test-label selection."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from data import load_data
from models import make_model
from structure import Structure
from benchmark import deployment_path, sync
from evaluation_core import select_configuration
from run_suite import job_folder


def heldout_data(cfg, root, smoke):
    if not smoke:
        return load_data(cfg, root, download=False)[2]
    gen = torch.Generator().manual_seed(777)
    c, size = (1, 28) if cfg['dataset'] == 'fashionmnist' else (3, 64 if cfg['dataset'] == 'tinyimagenet' else 32)
    torch.randn(240, c, size, size, generator=gen)  # Skip the same training tensor draw.
    return TensorDataset(torch.randn(240, c, size, size, generator=gen), torch.arange(240) % cfg['classes'])


def replay(folder, data, trace, device, fixed_removal=None):
    saved = torch.load(folder / 'deployment.pt', map_location='cpu', weights_only=False)
    cfg, method = saved['config'], saved['method']
    model = make_model(cfg['dataset'], exits=method == 'slex_style', switch_bn=method == 'slimmable').eval()
    model.load_state_dict(saved['model'])
    structure = Structure(model, cfg['min_active_ratio'], cfg['first_min_active_ratio'])
    profile = json.loads((folder / 'compact_profile.json').read_text())
    val = json.loads((folder / 'final_validation.json').read_text())['policy_accuracy_all_levels']
    split = json.loads((folder.parent / 'splits.json').read_text())
    candidates = [{'id': str(row['requested_removal']), 'validation_accuracy': val[str(row['requested_removal'])],
                   'costs': {'model_tensor_bytes': row['model_tensor_bytes']}}
                  for row in profile['operating_points'] if row['variant'] == 'global' and row['exit'] == 'final']
    if fixed_removal is not None:
        candidates = [c for c in candidates if c['id'] == str(fixed_removal)]
        if not candidates: raise ValueError('Fixed removal was not profiled')
    suffix = '' if fixed_removal is None else f'_fixed_{fixed_removal}'
    rows = []
    for cid, client in enumerate(split['deployment_clients']):
        active, selected_id = None, None
        for step, request in enumerate(trace['requests']):
            index = client['test'][step % len(client['test'])]
            x, target = data[index]
            budget = {'model_tensor_bytes': request['model_tensor_bytes']}
            selected = select_configuration(candidates, budget)
            sync(device)
            started = time.perf_counter()
            if selected is None:
                active, selected_id = None, None
                rows.append({'client': cid, 'step': step, 'sample_index': index, 'target': int(target),
                    'prediction': None, 'served': False, 'correct': False, 'selected_removal': None,
                    'budget_model_tensor_bytes': budget['model_tensor_bytes'], 'actual_model_tensor_bytes': 0,
                    'budget_violation': False, 'switched': False, 'construction_transfer_ms': 0., 'inference_ms': 0.,
                    'request_ms': (time.perf_counter() - started) * 1000, 'cuda_peak_allocated_bytes': None})
                continue
            changed = selected['id'] != selected_id
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            build_ms = 0.
            if changed:
                active = None  # Release old model before allocating next; full weights retained on CPU.
                start_build = time.perf_counter()
                level = int(selected['id'])
                active = deployment_path(structure.compact(model, saved['masks'][level], level), 'final').to(device).eval()
                sync(device)
                build_ms = (time.perf_counter() - start_build) * 1000
                selected_id = selected['id']
            x = x.unsqueeze(0).to(device)
            sync(device)
            infer_start = time.perf_counter()
            with torch.inference_mode():
                prediction = int(active(x).argmax(1))
            sync(device)
            infer_ms = (time.perf_counter() - infer_start) * 1000
            actual_bytes = sum(t.numel() * t.element_size() for t in active.state_dict().values())
            rows.append({'client': cid, 'step': step, 'sample_index': index, 'target': int(target),
                'prediction': prediction, 'served': True, 'correct': prediction == int(target),
                'selected_removal': int(selected_id), 'budget_model_tensor_bytes': budget['model_tensor_bytes'],
                'actual_model_tensor_bytes': actual_bytes, 'budget_violation': actual_bytes > budget['model_tensor_bytes'],
                'switched': changed, 'construction_transfer_ms': build_ms, 'inference_ms': infer_ms,
                'request_ms': (time.perf_counter() - started) * 1000,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None})
            del x
        del active
    with (folder / f'dynamic_requests{suffix}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {'fixed_removal': fixed_removal, 'trace': trace, 'device': str(device), 'variant': 'global', 'exit': 'final',
        'selection_data': 'training-pool validation; deployment targets only score served predictions',
        'trace_sha256': hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256((folder / 'deployment.pt').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notes': ['Synthetic resource trace, actual inference on held-out examples (synthetic examples only in smoke mode).',
                  'Memory budget covers model tensors, not total CPU/GPU process memory.',
                  'No latency deadline or energy budget. Timing includes observed rebuild/transfer and request inference.',
                  'Clients with fewer requests than trace length cycle their held-out indices; requests are not independent samples.',
                  'All methods use final exits and the same trace; no personalized dynamic-policy claim.']}
    (folder / f'dynamic_metadata{suffix}.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.suite)
    plan = json.loads((output / 'suite_plan.json').read_text())
    for dataset in plan['datasets']:
        jobs = [j for j in plan['jobs'] if j['category'] == 'main' and j['dataset'] == dataset]
        if not jobs: continue
        for seed in plan['seeds']:
            anchor_job = next((j for j in jobs if j['method'] == 'rasfl'), next((j for j in jobs if j['method'] == 'fedavg'), jobs[0]))
            anchor_folder = job_folder(output, anchor_job, seed, plan['mode'])
            profile = json.loads((anchor_folder / 'compact_profile.json').read_text())
            full = next(r['model_tensor_bytes'] for r in profile['operating_points']
                        if r['variant'] == 'global' and r['exit'] == 'final' and r['requested_removal'] == 0)
            fractions = [1.1, .8, .55, .3, .05, .55, .8, 1.1] * 2
            trace = {'name': 'declared_memory_sawtooth_v1', 'dataset': dataset, 'seed': seed,
                     'anchor': f"global {anchor_job['method']} full model tensor bytes", 'anchor_bytes': full,
                     'synthetic_resource_trace': True, 'synthetic_input_data': plan['synthetic'],
                     'requests': [{'fraction_of_anchor': f, 'model_tensor_bytes': int(f * full)} for f in fractions]}
            data = heldout_data(anchor_job['config'], args.data_root, plan['synthetic'])
            for job in jobs:
                folder = job_folder(output, job, seed, plan['mode'])
                print(f'Dynamic replay: {job["id"]}, seed={seed}', flush=True)
                replay(folder, data, trace, torch.device(args.device))
                if job['method'] == 'rasfl':
                    replay(folder, data, trace, torch.device(args.device), fixed_removal=0)
