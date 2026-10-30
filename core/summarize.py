"""Client means -> within-seed capacity means -> independent seed mean/sample SD."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(root):
    runs = defaultdict(list)
    seen = set()
    identities = {}
    for path in sorted(Path(root).rglob('clients.csv')):
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        if manifest['synthetic_smoke']:
            continue
        dataset, method, seed = manifest['config']['dataset'], manifest['method'], manifest['seed']
        key = dataset, method, seed
        if key in seen:
            raise ValueError(f'Duplicate run: {key}')
        seen.add(key)
        shared_key = dataset, seed
        identity = (manifest['split_sha256'], json.dumps(manifest['config'], sort_keys=True))
        if shared_key in identities and identities[shared_key] != identity:
            raise ValueError('Methods do not share the same splits/configuration')
        identities[shared_key] = identity
        groups = defaultdict(list)
        with path.open() as f:
            for row in csv.DictReader(f):
                groups[row['variant'], row['exit'], int(row['requested_removal'])].append(row)
        per_variant = defaultdict(dict)
        for (variant, exit, p), clients in groups.items():
            if {int(r['client']) for r in clients} != set(range(manifest['config']['deployment_clients'])) or len(clients) != manifest['config']['deployment_clients']:
                raise ValueError(f'Missing/duplicated deployment client: {path}')
            metrics = {m: statistics.mean(float(r[m]) for r in clients) for m in ('acc', 'macro_f1', 'balanced_acc')}
            per_variant[variant, exit][p] = metrics
            accs = sorted(float(r['acc']) for r in clients)
            runs[dataset, method, variant, exit, f'removal_{p}'].append({'seed': seed, **metrics,
                       'worst_client_acc': accs[0], 'median_client_acc': statistics.median(accs)})
        for (variant, exit), levels in per_variant.items():
            if method not in ('fedavg', 'fedprox'):
                if set(levels) != {0, 5, 10, 15, 20, 25, 30, 40}:
                    raise ValueError('Incomplete adaptive capacity evaluation')
                values = {m: statistics.mean(levels[p][m] for p in (5, 10, 15, 20, 25, 30, 40)) for m in ('acc', 'macro_f1', 'balanced_acc')}
                runs[dataset, method, variant, exit, 'capacity_mean'].append({'seed': seed, **values})
    result = []
    for key, values in sorted(runs.items()):
        result.append({'dataset': key[0], 'method': key[1], 'variant': key[2], 'exit': key[3], 'operating_point': key[4],
                       'n_seeds': len(values), 'complete_paper_seeds': {v['seed'] for v in values} == {42, 123, 2024},
                       'seed_values': values,
                       'mean': {m: statistics.mean(v[m] for v in values) for m in ('acc', 'macro_f1', 'balanced_acc')},
                       'sample_std': {m: statistics.stdev(v[m] for v in values) if len(values) > 1 else None for m in ('acc', 'macro_f1', 'balanced_acc')}})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output', default='summary.json')
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(summarize(args.root), indent=2))
