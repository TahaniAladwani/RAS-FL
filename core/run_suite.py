"""Plan/run the main comparisons, paper ablations, and declared sensitivity grid."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

METHODS = ['fedavg', 'fedprox', 'feddrop_style', 'slimmable', 'ofa_style', 'slex_style', 'rasfl']
DATASETS = ['fashionmnist', 'cifar10', 'cifar100']
ABLATIONS = {'random_masks': {'ordering': 'random'}, 'with_distillation': {'kd_alpha': 0.2},
             'no_shrink_training': {'mask_probability': 0.}, 'full_model_selection': {'full_validation_weight': 1.}}
SENSITIVITY = {'importance_beta': [.9, .95, .99], 'mask_probability': [.1, .25, .5],
               'masked_levels_per_batch': [1, 2, 4], 'full_validation_weight': [.25, .5, .75]}
DEFAULTS = {'ordering': 'importance', 'masked_levels_per_batch': 1, 'full_validation_weight': .5}


def make_plan(datasets, sections, seeds, mode, project):
    jobs = []
    for dataset in datasets:
        base = {**json.loads((project / 'configs' / f'{dataset}.json').read_text()), **DEFAULTS}
        for method in METHODS:
            if 'main' in sections:
                jobs.append({'id': f'main/{dataset}/{method}', 'category': 'main', 'dataset': dataset,
                             'method': method, 'config': dict(base), 'overrides': {}})
        if 'ablations' in sections:
            for name, overrides in ABLATIONS.items():
                jobs.append({'id': f'ablations/{dataset}/{name}', 'category': 'ablations',
                             'dataset': dataset, 'method': 'rasfl', 'config': {**base, **overrides}, 'overrides': overrides})
        if 'sensitivity' in sections:
            for parameter, values in SENSITIVITY.items():
                for value in values:
                    if value == base[parameter]:
                        continue  # Main RAS-FL is the reference value, not another independent run.
                    jobs.append({'id': f'sensitivity/{dataset}/{parameter}_{value}', 'category': 'sensitivity',
                                 'dataset': dataset, 'method': 'rasfl', 'config': {**base, parameter: value},
                                 'overrides': {parameter: value}})
    if 'sensitivity' in sections and not any(j['category'] == 'main' and j['method'] == 'rasfl' for j in jobs):
        for dataset in datasets:
            base = {**json.loads((project / 'configs' / f'{dataset}.json').read_text()), **DEFAULTS}
            jobs.append({'id': f'main/{dataset}/rasfl', 'category': 'main', 'dataset': dataset,
                         'method': 'rasfl', 'config': base, 'overrides': {}})
    for dataset in datasets:
        if 'levels' in sections:
            base = {**json.loads((project / 'configs' / f'{dataset}.json').read_text()), **DEFAULTS}
            for levels in ([10,40], [10,25,40], [5,10,20,30,40], [5,10,15,20,25,30,40]):
                overrides = {'training_removal_levels': levels}
                jobs.append({'id': f'levels/{dataset}/levels_{len(levels)}', 'category': 'levels',
                             'dataset': dataset, 'method': 'rasfl', 'config': {**base, **overrides}, 'overrides': overrides})
    return {'schema': 2, 'mode': mode, 'synthetic': mode == 'smoke', 'datasets': datasets,
            'seeds': seeds, 'sections': sections, 'training_clients': 20, 'deployment_clients': 20,
            'sensitivity_grid': SENSITIVITY, 'jobs': jobs,
            'notes': ['Checkpoint selection uses training-pool validation; deployment test scores are evaluation only.']}


def job_folder(output, job, seed, mode):
    return output / 'runs' / job['id'] / ('SMOKE_SYNTHETIC' if mode == 'smoke' else 'experiments') / job['dataset'] / f'seed_{seed}' / job['method']


def run(args):
    project = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    seeds = [42] if args.mode == 'smoke' else args.seeds
    plan = make_plan(args.datasets, args.sections, seeds, args.mode, project)
    if args.methods is not None:
        if args.sections != ['main']:
            raise ValueError('--methods requires --sections main')
        plan['jobs'] = [job for job in plan['jobs'] if job['method'] in args.methods]
    for job in plan['jobs']:
        if args.train_alpha is not None:
            job['config']['train_alpha'] = args.train_alpha
        job['config']['early_stopping_patience'] = args.early_stopping_patience
        job['config']['early_stopping_min_delta'] = args.early_stopping_min_delta
        if args.rounds is not None: job['config']['rounds'] = args.rounds
    plan['benchmark_requested'] = not args.no_benchmark
    plan['dynamic_requested'] = not args.no_dynamic
    plan['data_root'] = str(Path(args.data_root).resolve())
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'suite_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('An incompatible suite plan exists. Choose a fresh --output folder.')
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"Planned {len(plan['jobs']) * len(seeds)} method/seed runs; mode={args.mode}; output={output}", flush=True)
    if args.plan_only:
        return
    if args.mode == 'train' and 'tinyimagenet' in args.datasets:
        if not (Path(args.data_root) / 'tiny-imagenet-200/val/val_annotations.txt').exists():
            raise FileNotFoundError('Extract full Tiny-ImageNet before starting the four-dataset suite')
    status = {}
    configs = output / 'resolved_configs'
    configs.mkdir(exist_ok=True)
    def call(argv):
        print('Running: ' + ' '.join(map(str, argv)), flush=True)
        subprocess.run([sys.executable, *map(str, argv)], cwd=project, check=True)
    for number, job in enumerate(plan['jobs'], 1):
        config_path = configs / (job['id'].replace('/', '__') + '.json')
        config_path.write_text(json.dumps(job['config'], indent=2))
        command = [project / 'run.py', '--config', config_path, '--method', job['method'],
                   '--output', output / 'runs' / job['id'], '--data-root', Path(args.data_root).resolve(),
                   '--device', args.device, '--threads', args.threads, '--seeds', *seeds]
        if args.mode == 'smoke': command.append('--smoke')
        if args.download: command.append('--download')
        if args.resume: command.append('--resume')
        try:
            call(command)
            if job['category'] == 'main' and not args.no_benchmark:
                for seed in seeds:
                    folder = job_folder(output, job, seed, args.mode)
                    profile_file = folder / 'compact_profile.json'
                    checkpoint_sha = hashlib.sha256((folder / 'deployment.pt').read_bytes()).hexdigest()
                    valid = False
                    if args.resume and profile_file.exists():
                        old = json.loads(profile_file.read_text())
                        valid = old.get('checkpoint_sha256') == checkpoint_sha and old.get('benchmark_source_sha256') == hashlib.sha256((project / 'benchmark.py').read_bytes()).hexdigest()
                        if job['method'] == 'rasfl':
                            valid = valid and any(p['variant'] == 'personalized' for p in old['operating_points'])
                    if not valid:
                        command = [project / 'benchmark.py', folder / 'deployment.pt', '--output', profile_file,
                                   '--device', args.device, '--threads', args.threads,
                                   '--iterations', 3 if args.mode == 'smoke' else args.profile_iterations,
                                   '--warmup', 1 if args.mode == 'smoke' else 20]
                        if job['method'] == 'rasfl': command.append('--personalized')
                        call(command)
            status[job['id']] = {'status': 'complete', 'position': number}
            (output / 'suite_status.json').write_text(json.dumps(status, indent=2))
        except BaseException as exc:
            status[job['id']] = {'status': 'failed_or_interrupted', 'error': str(exc)}
            (output / 'suite_status.json').write_text(json.dumps(status, indent=2))
            raise
    if not args.no_dynamic and not args.no_benchmark and 'main' in args.sections:
        call([project / 'dynamic_replay.py', '--suite', output, '--data-root', Path(args.data_root).resolve(),
              '--device', args.device, '--threads', args.threads])
    call([project / 'export_results.py', '--suite', output])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=None)
    p.add_argument('--train-alpha', type=float, default=None)
    p.add_argument('--sections', nargs='+', choices=['main', 'ablations', 'sensitivity', 'levels'], default=['main', 'ablations', 'sensitivity'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2024])
    p.add_argument('--mode', choices=['smoke', 'train'], default='train')
    p.add_argument('--output', required=True)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--device', default='cpu')
    p.add_argument('--rounds', type=int, default=None)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--download', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--no-benchmark', action='store_true')
    p.add_argument('--no-dynamic', action='store_true')
    p.add_argument('--profile-iterations', type=int, default=100)
    p.add_argument('--early-stopping-patience', type=int, default=5)
    p.add_argument('--early-stopping-min-delta', type=float, default=0.)
    args = p.parse_args()
    if args.train_alpha is not None and not (0 < args.train_alpha < float('inf')):
        p.error('--train-alpha must be finite and positive')
    if args.methods is not None and len(set(args.methods)) != len(args.methods):
        p.error('Duplicate method requested')
    if args.early_stopping_patience < 1 or args.early_stopping_min_delta < 0:
        p.error('Invalid early-stopping settings')
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate dataset/seed requested')
    run(args)
