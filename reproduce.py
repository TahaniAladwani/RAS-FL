"""Run the dataset, concentration, and experiment grids."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('suite_plan', ROOT / 'core/run_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
DATASETS = suite.DATASETS
METHODS = suite.METHODS + ['fiarse', 'fjord']
SECTIONS = ['main', 'ablations', 'sensitivity', 'levels', 'baselines']


def commands(args):
    tasks = []
    seeds = [42] if args.smoke else args.seeds
    core_sections = [s for s in args.sections if s != 'baselines']
    for alpha in args.alphas:
        if core_sections and (args.methods is None or any(m in suite.METHODS for m in args.methods)):
            dest = args.output / f'alpha_{alpha:g}' / 'core'
            plan = suite.make_plan(args.datasets, core_sections, seeds, 'smoke' if args.smoke else 'train', ROOT / 'core')
            if args.methods:
                plan['jobs'] = [j for j in plan['jobs'] if j['method'] in args.methods]
            command = [sys.executable, str(ROOT / 'core/run_suite.py'), '--datasets', *args.datasets,
                       '--train-alpha', str(alpha), '--sections', *core_sections, '--seeds', *map(str, seeds),
                       '--output', str(dest), '--data-root', str(args.data_root), '--device', args.device,
                       '--threads', str(args.threads), '--early-stopping-patience', str(args.patience)]
            if args.methods: command += ['--methods', *[m for m in args.methods if m in suite.METHODS]]
            if args.smoke: command += ['--mode', 'smoke']
            if args.no_benchmark: command += ['--no-benchmark']
            if args.no_dynamic: command += ['--no-dynamic']
            if args.rounds is not None: command += ['--rounds', str(args.rounds)]
            tasks.append({'name': f'alpha_{alpha:g}/core', 'runs': len(plan['jobs']) * len(seeds), 'command': command})
        if 'baselines' in args.sections:
            methods = [m for m in ['fiarse', 'fjord'] if args.methods is None or m in args.methods]
            if not methods: continue
            for dataset in args.datasets:
                command = [sys.executable, str(ROOT / 'baselines' / dataset / 'run_experiments.py'),
                           '--alphas', str(alpha), '--methods', *methods, '--seeds', *map(str, seeds),
                           '--output', str(args.output / f'alpha_{alpha:g}' / 'baselines' / dataset),
                           '--data-root', str(args.data_root), '--device', args.device,
                           '--threads', str(args.threads), '--patience', str(args.patience)]
                if args.smoke: command += ['--smoke']
                if args.rounds is not None: command += ['--rounds', str(args.rounds)]
                tasks.append({'name': f'alpha_{alpha:g}/{dataset}/baselines', 'runs': len(methods) * len(seeds), 'command': command})
    for task in tasks:
        if args.download: task['command'].append('--download')
        if args.resume: task['command'].append('--resume')
    return tasks


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--alphas', nargs='+', type=float, choices=[.1, .4], default=[.1, .4])
    p.add_argument('--sections', nargs='+', choices=SECTIONS, default=SECTIONS)
    p.add_argument('--methods', nargs='+', choices=METHODS)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2024])
    p.add_argument('--output', type=Path, default=Path('results/paper'))
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--rounds', type=int)
    p.add_argument('--patience', type=int, default=5, help='Early-stopping patience shared by all methods')
    for name in ['download', 'resume', 'plan-only', 'smoke', 'no-benchmark', 'no-dynamic']:
        p.add_argument('--' + name, action='store_true')
    return p


def main():
    p = parser()
    args = p.parse_args()
    for name in ['datasets', 'alphas', 'sections', 'seeds', 'methods']:
        values = getattr(args, name)
        if values and len(values) != len(set(values)): p.error(f'Duplicate {name}')
    if args.methods and any(s not in ('main', 'baselines') for s in args.sections):
        p.error('--methods requires --sections main and/or baselines')
    if min(args.threads, args.patience, args.rounds or 1) < 1:
        p.error('Threads, rounds, and patience must be positive')
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.smoke: args.output /= 'SMOKE_SYNTHETIC'
    tasks = commands(args)
    if not tasks: p.error('No experiments match the requested sections and methods')
    plan = {'synthetic': args.smoke, 'datasets': args.datasets, 'alphas': args.alphas,
            'seeds': [42] if args.smoke else args.seeds, 'sections': args.sections,
            'training_runs': sum(t['runs'] for t in tasks), 'tasks': tasks}
    # Resume changes execution behavior, not experiment identity.
    for task in plan['tasks']:
        task['command'] = [v for v in task['command'] if v != '--resume']
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'experiment_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Output contains a different experiment plan; choose another --output.')
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(f"{plan['training_runs']} training runs; plan saved to {path}", flush=True)
    if args.plan_only: return
    for task in tasks:
        print(task['name'], flush=True)
        command = task['command'] + (['--resume'] if args.resume else [])
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'summarize.py'), str(args.output)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
