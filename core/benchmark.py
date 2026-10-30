"""Static global and client-specific compact inference measurements."""
import argparse
import gzip
import hashlib
import io
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models import make_model
from structure import Structure
from run import check_compaction


def deployment_path(small, exit_index):
    if exit_index == 'final':
        small.exits = nn.ModuleList()
    else:
        small.stages = nn.ModuleList(list(small.stages)[:exit_index + 1])
        small.exits = nn.ModuleList([nn.Identity() for _ in range(exit_index)] + [small.exits[exit_index]])
        small.head = nn.Identity()
    return small


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(small, x_cpu, device, exit_index, iterations, warmup):
    state_buffer = io.BytesIO()
    torch.save(small.state_dict(), state_buffer)
    serialized_bytes = state_buffer.tell()
    del state_buffer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    small = small.to(device)
    x = x_cpu.to(device)
    macs, hooks = [0], []
    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            macs[0] += output.numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        else:
            macs[0] += output.numel() * module.in_features
    for module in small.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count))
    del module
    with torch.inference_mode():
        small.forward_exit(x, exit_index)
    for hook in hooks:
        hook.remove()
    with torch.inference_mode():
        for _ in range(warmup):
            small.forward_exit(x, exit_index)
        sync(device)
        times = []
        for _ in range(iterations):
            sync(device)
            start = time.perf_counter()
            small.forward_exit(x, exit_index)
            sync(device)
            times.append((time.perf_counter() - start) * 1000)
    batch_size = len(x_cpu)
    return {'batch_size': batch_size, 'latency_p50_ms': float(np.percentile(times, 50)),
            'latency_p95_ms': float(np.percentile(times, 95)), 'latency_mean_ms': float(np.mean(times)),
            'latency_samples_ms': times, 'conv_linear_macs_per_batch': macs[0],
            'conv_linear_macs_per_sample': macs[0] / batch_size,
            'conv_linear_flops_per_sample': 2 * macs[0] / batch_size,
            'parameter_count': sum(p.numel() for p in small.parameters()),
            'model_tensor_bytes': sum(t.numel() * t.element_size() for t in small.state_dict().values()),
            'state_dict_serialized_bytes': serialized_bytes,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None}


def profile(checkpoint, device, batch_size=1, iterations=100, warmup=20, personalized=None):
    cfg, method = checkpoint['config'], checkpoint['method']
    model = make_model(cfg['dataset'], exits=method == 'slex_style', switch_bn=method == 'slimmable')
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Full retained model stays on CPU, separate from measured compact model.
    structure = Structure(model, cfg['min_active_ratio'], cfg['first_min_active_ratio'])
    channels, size = (1, 28) if cfg['dataset'] == 'fashionmnist' else (3, 64 if cfg['dataset'] == 'tinyimagenet' else 32)
    x_cpu = torch.randn(batch_size, channels, size, size, generator=torch.Generator().manual_seed(882))
    levels = [0] if method in ('fedavg', 'fedprox') else sorted(checkpoint['masks'])
    exits = list(range(len(model.exits))) + ['final'] if method == 'slex_style' else ['final']
    variants = [('global', -1, checkpoint['masks'])]
    for cid, roadmaps in (personalized or {}).items():
        variants.append(('personalized', int(cid), {int(p): {g: torch.tensor(m, dtype=torch.bool) for g, m in masks.items()}
                                                   for p, masks in roadmaps.items()}))
    records = []
    for variant, cid, masks_by_level in variants:
        for level in levels:
            for exit_index in exits:
                small = structure.compact(model, masks_by_level[level], level)
                error = check_compaction(model, small, structure, masks_by_level[level], x_cpu, level, exit_index)
                small = deployment_path(small, exit_index)
                record = measure(small, x_cpu, device, exit_index, iterations, warmup)
                records.append({'variant': variant, 'client': cid, 'requested_removal': level,
                                'exit': str(exit_index), 'compact_logit_max_error': error, **record})
                del small
    for row in records:
        full = next(r for r in records if r['variant'] == row['variant'] and r['client'] == row['client']
                    and r['exit'] == row['exit'] and r['requested_removal'] == 0)
        for key in ('parameter_count', 'model_tensor_bytes', 'state_dict_serialized_bytes',
                    'conv_linear_flops_per_sample', 'cuda_peak_allocated_bytes'):
            row[key + '_reduction_percent'] = 100 * (1 - row[key] / full[key]) if full[key] else None
    return {'method': method, 'dataset': cfg['dataset'], 'seed': checkpoint['seed'],
            'device': str(device), 'precision': 'float32', 'warmup': warmup, 'iterations': iterations,
            'environment': {'platform': platform.platform(), 'processor': platform.processor(),
                'torch': torch.__version__, 'threads': torch.get_num_threads(), 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
            'retained_full_checkpoint_tensor_bytes_on_cpu': sum(t.numel() * t.element_size() for t in model.state_dict().values()),
            'notes': 'Static operating points. FLOPs=2*Conv/Linear MACs, excluding BN/activations/pooling. CPU RSS and energy not measured. Latency samples are not independent training seeds.',
            'operating_points': records}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--output', default='profile.json')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--personalized', action='store_true', help='Also profile saved per-client RAS-FL roadmaps')
    args = p.parse_args()
    if min(args.batch_size, args.iterations, args.warmup) < 1:
        p.error('Batch size, iterations and warmup must be positive')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    personalized = None
    if args.personalized:
        with gzip.open(Path(args.checkpoint).parent / 'personalized_roadmaps.json.gz', 'rt') as handle:
            personalized = json.load(handle)
    result = profile(saved, torch.device(args.device), args.batch_size, args.iterations, args.warmup, personalized)
    result['checkpoint_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    result['benchmark_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(result, indent=2))
