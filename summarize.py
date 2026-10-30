"""Summarize completed runs without training or loading model checkpoints."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

LIMITS = {'fashionmnist': 40, 'cifar10': 30, 'cifar100': 20}
NAMES = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'feddrop_style': 'FedDropout-style',
         'slimmable': 'Slimmable-style', 'ofa_style': 'OFA-style', 'slex_style': 'SLEX-style',
         'rasfl': 'RAS-FL', 'fiarse': 'FIARSE', 'fjord': 'FjORD'}
KEYS = ['dataset', 'alpha', 'experiment', 'method', 'variant', 'synthetic']
METRICS = ['full_acc', 'runtime_avg', 'acc_10', 'acc_20', 'acc_30', 'acc_40',
           'balanced_acc', 'test_loss', 'checkpoint_score', 'selected_round']


def read_csv(path):
    with path.open() as handle: return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def core_rows(root):
    rows = []
    for path in root.glob('alpha_*/core/suite_plan.json'):
        plan = json.loads(path.read_text())
        for job in plan['jobs']:
            for seed in plan['seeds']:
                folder = path.parent / 'runs' / job['id'] / ('SMOKE_SYNTHETIC' if plan['synthetic'] else 'experiments') / job['dataset'] / f'seed_{seed}' / job['method']
                manifest_path = folder / 'manifest.json'
                if not manifest_path.exists(): continue
                manifest = json.loads(manifest_path.read_text())
                if not manifest.get('evaluation_complete'): continue
                validation = json.loads((folder / 'final_validation.json').read_text())
                clients = [r for r in read_csv(folder / 'clients.csv') if r['exit'] == 'final']
                for variant in sorted({r['variant'] for r in clients}):
                    bylevel = defaultdict(list)
                    for r in clients:
                        if r['variant'] == variant: bylevel[int(r['requested_removal'])].append(r)
                    for group in bylevel.values():
                        if {int(r['client']) for r in group} != set(range(20)) or len(group) != 20:
                            raise ValueError(f'Invalid client coverage: {folder}')
                    acc = {level: 100*statistics.mean(float(r['acc']) for r in group) for level, group in bylevel.items()}
                    reduced = [v for level,v in acc.items() if 0 < level <= LIMITS[job['dataset']]]
                    rows.append({**dict(zip(KEYS, [job['dataset'], job['config']['train_alpha'], job['id'], NAMES[job['method']], variant, plan['synthetic']])),
                        'seed': seed, 'full_acc': acc[0], 'runtime_avg': statistics.mean(reduced) if reduced else None,
                        **{f'acc_{level}': acc.get(level) for level in [10,20,30,40]},
                        'balanced_acc': 100*statistics.mean(float(r['balanced_acc']) for r in bylevel[0]),
                        'test_loss': statistics.mean(float(r['loss']) for r in bylevel[0]),
                        'checkpoint_score': 100*validation['selection_before_stabilization'],
                        'selected_round': validation['selected_round']})
    return rows


def baseline_rows(root):
    rows = []
    for path in root.glob('alpha_*/baselines/**/results.csv'):
        values = read_csv(path)
        if not values: continue
        first = values[0]
        dataset = first['dataset']
        acc = {int(r['removal_reference_pct']): float(r['equal_client_accuracy_pct']) for r in values}
        if set(acc) != {0,5,10,15,20,25,30,40} or len(values) != 8:
            raise ValueError(f'Incomplete capacity sweep: {path}')
        reduced = [v for level,v in acc.items() if 0 < level <= LIMITS[dataset]]
        rows.append({**dict(zip(KEYS, [dataset, float(first['train_alpha']), f"main/{dataset}/{first['method']}", NAMES[first['method']], 'global', first['synthetic_smoke'] == 'True'])),
                     'seed': int(first['seed']), 'full_acc': acc[0], 'runtime_avg': statistics.mean(reduced),
                     **{f'acc_{level}': acc[level] for level in [10,20,30,40]},
                     'balanced_acc': None, 'test_loss': None, 'checkpoint_score': None,
                     'selected_round': float(first['selected_round'])})
    return rows


def summarize(root):
    rows = core_rows(root) + baseline_rows(root)
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in KEYS)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        seeds = [r['seed'] for r in group]
        if len(seeds) != len(set(seeds)): raise ValueError(f'Duplicate seed: {key}')
        entry = {**dict(zip(KEYS,key)), 'runs': len(seeds), 'seeds': '|'.join(map(str,sorted(seeds)))}
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            entry[metric+'_mean'] = statistics.mean(values) if values else None
            entry[metric+'_sample_sd'] = statistics.stdev(values) if len(values)>1 else None
            entry[metric+'_runs'] = len(values)
        summaries.append(entry)
    write_csv(root/'per_seed.csv', rows)
    write_csv(root/'summary.csv', summaries)
    print(f'Summarized {len(rows)} completed seed/variant records in {root}')
    return summaries


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    summarize(p.parse_args().output.resolve())
