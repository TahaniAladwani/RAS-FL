"""Build auditable paper tables, graph data, figures, and one portable results ZIP."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from run_suite import job_folder, METHODS, DATASETS, SENSITIVITY

METRICS = ['acc', 'macro_f1', 'balanced_acc', 'loss', 'compact_path_parameters',
           'active_output_structures', 'total_output_structures', 'active_output_structure_ratio',
           'achieved_group_removal', 'compact_logit_max_error']
HARDWARE = ['latency_p50_ms', 'latency_p95_ms', 'latency_mean_ms', 'conv_linear_macs_per_sample',
            'conv_linear_flops_per_sample', 'parameter_count', 'model_tensor_bytes',
            'state_dict_serialized_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes']
HARDWARE += [k + '_reduction_percent' for k in ['parameter_count', 'model_tensor_bytes',
              'state_dict_serialized_bytes', 'conv_linear_flops_per_sample', 'cuda_peak_allocated_bytes']]
KEYS = ['job_id', 'category', 'dataset', 'method', 'variant', 'exit', 'operating_point']


def csv_read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as handle:
        if keys:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def number(value):
    if value in ('', None): return None
    value = float(value)
    if not math.isfinite(value): raise ValueError('Nonfinite numeric result')
    return value


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        seeds = [int(r['seed']) for r in group]
        if len(seeds) != len(set(seeds)): raise ValueError(f'Duplicate seed in aggregate {key}')
        entry = {**dict(zip(keys, key)), 'n_seeds': len(seeds), 'seeds': '|'.join(map(str, sorted(seeds))),
                 'synthetic': group[0]['synthetic']}
        for metric in metrics:
            values = [r.get(metric) for r in group if r.get(metric) is not None]
            entry[metric + '_mean'] = mean(values)
            entry[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else None
            entry[metric + '_n_seeds'] = len(values)
        result.append(entry)
    return result


def tex_escape(s):
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def latex_table(path, rows, columns, synthetic):
    lines = ['% SYNTHETIC SMOKE -- NOT PAPER RESULTS' if synthetic else '% Generated from recorded independent-seed summaries.',
             '% Uses longtable; add \\usepackage{longtable} to the manuscript.',
             r'\begin{longtable}{' + 'l' * len(columns) + '}',
             ' & '.join(tex_escape(title) for _, title in columns) + r' \\ \hline']
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float): value = f'{value:.4f}'
            values.append('--' if value is None else tex_escape(value))
        lines.append(' & '.join(values) + r' \\')
    lines.append(r'\end{longtable}')
    path.write_text('\n'.join(lines) + '\n')


def figures(output, curves, convergence, sensitivity, ablations, dynamic, synthetic):
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 130, 'savefig.bbox': 'tight'})
    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    files = []
    prefix = 'SYNTHETIC SMOKE — NOT RESEARCH RESULTS\n' if synthetic else ''
    def save(fig, name):
        for suffix in ('png', 'pdf'):
            file = folder / f'{name}.{suffix}'
            fig.savefig(file)
            files.append(str(file.relative_to(output)))
        plt.close(fig)
    for dataset in sorted({r['dataset'] for r in curves}):
        main = [r for r in curves if r['dataset'] == dataset and r['category'] == 'main' and r['exit'] == 'final'
                and r['operating_point'].startswith('removal_')]
        for xkey, xlabel in [('removal', 'Requested group removal (%)'),
            ('compact_path_parameters', 'Active path parameters'),
            ('active_output_structure_ratio', 'Active output-structure ratio (includes protected head)'),
            ('conv_linear_flops_per_sample', 'Conv/Linear FLOPs per example (2 × MACs)'),
            ('latency_mean_ms', 'Measured mean latency (ms)'), ('model_tensor_bytes', 'Compact model tensor bytes')]:
            fig, ax = plt.subplots(figsize=(7.2, 4.3))
            count = 0
            for method, variant in sorted({(r['method'], r['variant']) for r in main}):
                points = [r for r in main if (r['method'], r['variant']) == (method, variant)]
                points.sort(key=lambda r: int(r['operating_point'].split('_')[1]))
                points = [r for r in points if xkey == 'removal' or r.get(xkey + '_mean') is not None]
                if not points: continue
                xs = [int(r['operating_point'].split('_')[1]) if xkey == 'removal' else r[xkey + '_mean'] for r in points]
                ys = [100 * r['acc_mean'] for r in points]
                ax.errorbar(xs, ys, yerr=[100 * (r['acc_std'] or 0) for r in points],
                    marker='o', markersize=3, linewidth=1, capsize=2, label=f'{method} / {variant}')
                count += 1
            if count:
                ax.set(xlabel=xlabel, ylabel='Client-mean accuracy (%)', title=prefix + dataset)
                ax.grid(alpha=.2)
                ax.legend(fontsize=6.5, loc='best')
                save(fig, f'{dataset}_accuracy_vs_{xkey}')
            else: plt.close(fig)
        for metric in ('macro_f1', 'balanced_acc', 'loss'):
            fig, ax = plt.subplots(figsize=(7.2, 4.3))
            for method, variant in sorted({(r['method'], r['variant']) for r in main}):
                points = sorted([r for r in main if (r['method'], r['variant']) == (method, variant)],
                                key=lambda r: int(r['operating_point'].split('_')[1]))
                ax.errorbar([int(r['operating_point'].split('_')[1]) for r in points], [r[metric + '_mean'] for r in points],
                            yerr=[r[metric + '_std'] or 0 for r in points], marker='o', markersize=3, label=f'{method}/{variant}')
            ax.set(xlabel='Requested removal (%)', ylabel=metric, title=prefix + dataset)
            ax.legend(fontsize=6.5); ax.grid(alpha=.2)
            save(fig, f'{dataset}_{metric}_vs_removal')
        for metric in ('validation_full_accuracy', 'validation_full_loss', 'selection_score', 'retained_set_jaccard_at_40_previous_round'):
            points = [r for r in convergence if r['dataset'] == dataset and r['category'] == 'main']
            if not points: continue
            fig, ax = plt.subplots(figsize=(7.2, 4.3))
            for method in sorted({r['method'] for r in points}):
                values = sorted([r for r in points if r['method'] == method], key=lambda r: r['round'])
                ax.plot([r['round'] for r in values], [r[metric + '_mean'] for r in values], label=method)
            ax.set(xlabel='Communication round', ylabel=metric, title=prefix + dataset)
            ax.legend(fontsize=7); ax.grid(alpha=.2)
            save(fig, f'{dataset}_convergence_{metric}')
        for parameter in SENSITIVITY:
            points = sorted([r for r in sensitivity if r['dataset'] == dataset and r['parameter'] == parameter], key=lambda r: r['value'])
            if not points: continue
            fig, ax = plt.subplots(figsize=(6.2, 4))
            ax.errorbar([r['value'] for r in points], [100 * r['acc_mean'] for r in points],
                        yerr=[100 * (r['acc_std'] or 0) for r in points], marker='o', capsize=3)
            ax.set(xlabel=parameter, ylabel='Mean accuracy across evaluation capacities (%)', title=prefix + dataset)
            ax.grid(alpha=.2)
            save(fig, f'{dataset}_sensitivity_{parameter}')
        points = [r for r in ablations if r['dataset'] == dataset]
        if points:
            fig, ax = plt.subplots(figsize=(7.2, 4.3))
            for job in sorted({r['job_id'] for r in points}):
                values = sorted([r for r in points if r['job_id'] == job], key=lambda r: int(r['operating_point'].split('_')[1]))
                ax.errorbar([int(r['operating_point'].split('_')[1]) for r in values], [100 * r['acc_mean'] for r in values],
                            yerr=[100 * (r['acc_std'] or 0) for r in values], marker='o', label=job.split('/')[-1])
            ax.set(xlabel='Requested removal (%)', ylabel='Accuracy (%)', title=prefix + dataset + ' ablations')
            ax.legend(fontsize=7); ax.grid(alpha=.2)
            save(fig, f'{dataset}_ablations')
        points = [r for r in dynamic if r['dataset'] == dataset]
        if points:
            fig, ax = plt.subplots(figsize=(7.2, 4.3))
            ax.bar([r['method'] for r in points], [100 * r['served_fraction_mean'] for r in points])
            ax.tick_params(axis='x', rotation=30)
            ax.set(ylabel='Served requests (%)', title=prefix + dataset + ' declared memory trace')
            save(fig, f'{dataset}_dynamic_coverage')
    return files


def export(suite, include_checkpoints=False, no_figures=False):
    suite = Path(suite).resolve()
    project = Path(__file__).resolve().parent
    plan = json.loads((suite / 'suite_plan.json').read_text())
    output = suite / 'paper_results'
    if output.exists():
        # Rebuild generated report directory only; preserve source runs and prior ZIPs.
        for path in output.rglob('*'):
            if path.is_file() and path.suffix in ('.csv', '.json', '.md', '.tex', '.png', '.pdf'):
                path.unlink()
    output.mkdir(exist_ok=True)
    all_clients, seed_rows, histories, dynamic_seed = [], [], [], []
    inventory, raw_files, failures, profile_missing, dynamic_missing = [], [], [], [], []
    partial_profile_points = set()
    split_identity = {}
    for job in plan['jobs']:
        for seed in plan['seeds']:
            folder = job_folder(suite, job, seed, plan['mode'])
            name = f"{job['id']}/seed_{seed}"
            required = ['manifest.json', 'clients.csv', 'history.json', 'final_validation.json', 'predictions.jsonl.gz']
            missing = [f for f in required if not (folder / f).exists()]
            if missing:
                failures.append({'run': name, 'missing': missing}); continue
            manifest = json.loads((folder / 'manifest.json').read_text())
            if not manifest.get('evaluation_complete'):
                failures.append({'run': name, 'missing': ['evaluation_complete']}); continue
            if manifest['synthetic_smoke'] != plan['synthetic'] or manifest['seed'] != seed or manifest['method'] != job['method']:
                raise ValueError(f'Run identity mismatch: {name}')
            if manifest['config']['dataset'] != job['dataset']:
                raise ValueError('Dataset identity mismatch')
            if (folder / 'deployment.pt').exists() and hashlib.sha256((folder / 'deployment.pt').read_bytes()).hexdigest() != manifest['deployment_checkpoint_sha256']:
                raise ValueError(f'Checkpoint content hash mismatch: {name}')
            key = job['dataset'], seed
            if key in split_identity and split_identity[key] != manifest['split_sha256']:
                raise ValueError(f'Unpaired splits across experiments: {key}')
            split_identity[key] = manifest['split_sha256']
            prefix = {'job_id': job['id'], 'category': job['category'], 'dataset': job['dataset'],
                      'method': job['method'], 'seed': seed, 'synthetic': plan['synthetic']}
            clients = csv_read(folder / 'clients.csv')
            hardware = {}
            if (folder / 'compact_profile.json').exists():
                profile = json.loads((folder / 'compact_profile.json').read_text())
                if profile.get('checkpoint_sha256') != manifest['deployment_checkpoint_sha256']:
                    raise ValueError(f'Hardware profile checkpoint mismatch: {name}')
                for r in profile['operating_points']:
                    hardware[r['variant'], int(r['client']), str(r['exit']), int(r['requested_removal'])] = r
            elif job['category'] == 'main':
                profile_missing.append(name)
            grouped = defaultdict(list)
            for client in clients:
                row = {**prefix, **client}
                row['seed'] = seed
                row['synthetic'] = plan['synthetic']
                for metric in METRICS: row[metric] = number(row.get(metric))
                row['client'], row['requested_removal'] = int(row['client']), int(row['requested_removal'])
                prof = hardware.get((row['variant'], row['client'] if row['variant'] == 'personalized' else -1,
                                     row['exit'], row['requested_removal']))
                if prof is None and job['category'] == 'main':
                    partial_profile_points.add((name, row['variant'], row['exit'], row['requested_removal']))
                for metric in HARDWARE: row[metric] = number(prof.get(metric)) if prof else None
                all_clients.append(row)
                grouped[row['variant'], row['exit'], row['requested_removal']].append(row)
            per_variant = defaultdict(dict)
            for (variant, exit, level), values in grouped.items():
                if len(values) != 20 or {r['client'] for r in values} != set(range(20)):
                    raise ValueError(f'Invalid client coverage: {name}')
                metrics = {m: mean([r.get(m) for r in values]) for m in METRICS + HARDWARE}
                accs = sorted(r['acc'] for r in values)
                metrics.update(worst_client_accuracy=accs[0], median_client_accuracy=statistics.median(accs),
                               lower_decile_client_accuracy=float(__import__('numpy').quantile(accs, .1)))
                entry = {**prefix, 'variant': variant, 'exit': exit, 'operating_point': f'removal_{level}', **metrics}
                seed_rows.append(entry)
                per_variant[variant, exit][level] = entry
            for (variant, exit), points in per_variant.items():
                expected = {0} if job['method'] in ('fedavg', 'fedprox') else {0, 5, 10, 15, 20, 25, 30, 40}
                if set(points) != expected: raise ValueError(f'Incomplete capacity sweep: {name}')
                if len(expected) > 1:
                    seed_rows.append({**prefix, 'variant': variant, 'exit': exit, 'operating_point': 'capacity_mean',
                                      **{m: mean([points[p][m] for p in [p for p in sorted(expected - {0}) if p <= {'fashionmnist': 40, 'cifar10': 30, 'cifar100': 20}[job['dataset']]]]) for m in METRICS + HARDWARE},
                                      **{m: mean([points[p][m] for p in [p for p in sorted(expected - {0}) if p <= {'fashionmnist': 40, 'cifar10': 30, 'cifar100': 20}[job['dataset']]]]) for m in
                                         ['worst_client_accuracy', 'median_client_accuracy', 'lower_decile_client_accuracy']}})
            for row in json.loads((folder / 'history.json').read_text()):
                histories.append({**prefix, 'round': row['round'], 'selection_score': row['selection_score'],
                    'validation_full_accuracy': row['validation_accuracy']['0'],
                    'validation_full_loss': row['validation_metrics']['0']['loss'],
                    **{k: row[k] for k in ['round_wall_seconds', 'mean_objective_over_student_passes',
                        'retained_set_jaccard_at_40_previous_round', 'student_forward_passes', 'optimizer_steps']}})
            if (folder / 'dynamic_requests.csv').exists():
                dynamic_meta = json.loads((folder / 'dynamic_metadata.json').read_text())
                if dynamic_meta['checkpoint_sha256'] != manifest['deployment_checkpoint_sha256']:
                    raise ValueError(f'Dynamic replay checkpoint mismatch: {name}')
                requests = csv_read(folder / 'dynamic_requests.csv')
                by_client = defaultdict(list)
                for r in requests: by_client[int(r['client'])].append(r)
                client_results = []
                for cid, group in by_client.items():
                    served = [r for r in group if r['served'] == 'True']
                    switches = [float(r['construction_transfer_ms']) for r in group if r['switched'] == 'True']
                    correct = sum(r['correct'] == 'True' for r in served)
                    client_results.append({'served_fraction': len(served) / len(group), 'effective_accuracy': correct / len(group),
                        'served_accuracy': correct / len(served) if served else None,
                        'budget_violation_fraction': sum(r['budget_violation'] == 'True' for r in group) / len(group),
                        'switch_count': len(switches), 'mean_request_ms': mean([float(r['request_ms']) for r in group]),
                        'mean_switch_ms': mean(switches)})
                dynamic_seed.append({**prefix, **{k: mean([r[k] for r in client_results]) for k in client_results[0]}})
            elif job['category'] == 'main':
                dynamic_missing.append(name)
            inventory.append({**prefix, 'run': name, 'checkpoint_sha256': manifest['deployment_checkpoint_sha256'],
                              'split_sha256': manifest['split_sha256'], 'has_hardware_profile': bool(hardware),
                              'has_dynamic_trace': (folder / 'dynamic_requests.csv').exists()})
            for path in folder.iterdir():
                if path.suffix in ('.json', '.csv', '.gz') or (include_checkpoints and path.name == 'deployment.pt'):
                    raw_files.append((path, f'raw/{name}/{path.name}'))
            raw_files.append((folder.parent / 'splits.json', f'raw/{name}/splits.json'))
    metric_names = METRICS + HARDWARE + ['worst_client_accuracy', 'median_client_accuracy', 'lower_decile_client_accuracy']
    curves = aggregate(seed_rows, KEYS, metric_names)
    convergence = aggregate(histories, ['job_id', 'category', 'dataset', 'method', 'round'],
        ['selection_score', 'validation_full_accuracy', 'validation_full_loss', 'round_wall_seconds',
         'mean_objective_over_student_passes', 'retained_set_jaccard_at_40_previous_round', 'student_forward_passes', 'optimizer_steps'])
    dynamic = aggregate(dynamic_seed, ['job_id', 'category', 'dataset', 'method'],
                        ['served_fraction', 'effective_accuracy', 'served_accuracy', 'budget_violation_fraction',
                         'switch_count', 'mean_request_ms', 'mean_switch_ms'])
    main = [r for r in curves if r['category'] == 'main']
    overall = [r for r in main if r['exit'] == 'final' and r['operating_point'] in ('removal_0', 'capacity_mean')]
    hardware_global = [r for r in main if r['method'] == 'rasfl' and r['variant'] == 'global' and r['operating_point'].startswith('removal_')]
    hardware_personal = [r for r in main if r['method'] == 'rasfl' and r['variant'] == 'personalized' and r['operating_point'].startswith('removal_')]
    ablations = [r for r in curves if (r['category'] == 'ablations' or (r['category'] == 'main' and r['method'] == 'rasfl'))
                 and r['variant'] == 'global' and r['exit'] == 'final' and r['operating_point'].startswith('removal_')]
    sensitivity = []
    job_by_id = {j['id']: j for j in plan['jobs']}
    for row in curves:
        if row['method'] != 'rasfl' or row['variant'] != 'global' or row['exit'] != 'final' or row['operating_point'] != 'capacity_mean': continue
        job = job_by_id[row['job_id']]
        if row['category'] == 'main':
            for parameter in SENSITIVITY:
                sensitivity.append({**row, 'parameter': parameter, 'value': job['config'][parameter]})
        elif row['category'] == 'sensitivity':
            parameter = next(iter(job['overrides']))
            sensitivity.append({**row, 'parameter': parameter, 'value': job['overrides'][parameter]})
    outputs = {'all_client_results.csv': all_clients, 'all_seed_results.csv': seed_rows,
        'all_seed_summaries.csv': curves, 'table_01_overall.csv': overall,
        'table_02_global_compaction.csv': hardware_global, 'table_03_ablations.csv': ablations,
        'appendix_01_all_datasets_parameters_flops.csv': main,
        'appendix_02_all_capacities_all_exits.csv': main,
        'appendix_03_personalized_compaction.csv': hardware_personal,
        'appendix_04_extended_ablations.csv': ablations, 'appendix_sensitivity.csv': sensitivity,
        'convergence_seed_data.csv': histories, 'convergence_summary.csv': convergence,
        'dynamic_seed_data.csv': dynamic_seed, 'dynamic_summary.csv': dynamic,
        'run_inventory.csv': inventory}
    for name, values in outputs.items(): csv_write(output / 'tables' / name, values)
    # All figure data is also in all_seed_summaries / convergence_summary / sensitivity / dynamic_summary.
    tex_columns = [('dataset', 'Dataset'), ('method', 'Method'), ('variant', 'Variant'),
                   ('operating_point', 'Capacity'), ('acc_mean', 'Accuracy mean'), ('acc_std', 'Seed SD')]
    for name, values in [('table_01_overall', overall), ('table_03_ablations', ablations), ('appendix_all_capacities', main)]:
        columns = [('job_id', 'Experiment')] + tex_columns if 'ablation' in name else tex_columns
        latex_table(output / 'tables' / (name + '.tex'), values, columns, plan['synthetic'])
    png_files = [] if no_figures else figures(output, curves, convergence, sensitivity, ablations, dynamic, plan['synthetic'])
    full_plan = (set(plan['datasets']) == set(DATASETS) and set(plan['seeds']) == {42, 123, 2024}
                 and set(plan['sections']) == {'main', 'ablations', 'sensitivity'})
    coverage = {'schema': 2, 'synthetic': plan['synthetic'], 'expected_method_seed_runs': len(plan['jobs']) * len(plan['seeds']),
                'completed_method_seed_runs': len(inventory), 'missing_runs': failures, 'main_profiles_missing': profile_missing,
                'main_profile_points_missing': sorted(partial_profile_points), 'main_dynamic_replays_missing': dynamic_missing,
                'gpu_memory_measurements_available': any(r.get('cuda_peak_allocated_bytes_mean') is not None for r in main),
                'requested_plan_training_complete': not failures, 'full_three_dataset_three_seed_plan': full_plan,
                'paper_results': not plan['synthetic'], 'figures_created': png_files,
                'appendix_source_available': False,
                'limitations': ['Experiment settings are recorded in suite_plan.json.',
                    'The dynamic memory trace is synthetic; the workload and trace are recorded separately.',
                    'CPU runs have no GPU memory measurements; missing values are null, never zero.',
                    'Energy and peak host RSS are not measured. Static benchmarks do not measure dynamic switching; see separate replay.',
                    'The architecture schematic is conceptual and has no numeric experiment data to replace.',
                    'No claim of independent-seed results is supported by smoke outputs.'],
                'paper_mapping': {'main_table_1': 'tables/table_01_overall.csv', 'main_table_2': 'tables/table_02_global_compaction.csv',
                    'main_table_3': 'tables/table_03_ablations.csv', 'main_EX9_accuracy_parameters': 'figures/*accuracy_vs_compact_path_parameters.*',
                    'main_EX1_accuracy_structures': 'figures/*accuracy_vs_active_output_structure_ratio.*',
                    'appendix_table_1': 'tables/appendix_01_all_datasets_parameters_flops.csv',
                    'appendix_table_2': 'tables/appendix_02_all_capacities_all_exits.csv',
                    'appendix_table_3': 'tables/appendix_03_personalized_compaction.csv',
                    'appendix_table_4': 'tables/appendix_04_extended_ablations.csv',
                    'appendix_sensitivity_table_6_referenced': 'tables/appendix_sensitivity.csv',
                    'appendix_flop_figure': 'figures/*accuracy_vs_conv_linear_flops_per_sample.*',
                    'appendix_convergence': 'figures/*convergence*', 'dynamic_budget_experiment': 'tables/dynamic_summary.csv'}}
    (output / 'coverage.json').write_text(json.dumps(coverage, indent=2))
    (output / 'README_RESULTS.md').write_text('''# Results bundle for updating the paper

Start with coverage.json: it identifies completed runs, missing measurements, and source-to-table mapping.

The tables use fractions (accuracy 0.8 = 80%); latency is milliseconds; memory is bytes;
FLOPs count two operations per Conv/Linear MAC and exclude normalization/activations/pooling.
Means average clients uniformly, then capacities within seed where applicable, then training seeds.
Standard deviations are sample SDs between seed means. Null means unavailable; a single seed has null SD.
Global hardware rows are repeated across clients only for joining to per-client accuracies, then averaged
within a seed; timing samples are never counted as independent training seeds.

all_client_results.csv, all_seed_results.csv, and raw/ preserve the underlying results for replotting.
Predictions include source-pool sample indices, targets and predictions, so F1 can be verified independently.
All masks, adaptation records, split indices, resolved configs, software/source hashes, and benchmark
metadata are included. Model checkpoints remain on your training machine unless explicitly included.
Keep checkpoints for subsequent evaluation.

Figures are PNG and PDF; their underlying numbers are in tables/. Generated LaTeX tables require longtable.
Sensitivity and ablation grids must be reported as specified in suite_plan.json; do not select hyperparameters
using the final test table. Unavailable measurements remain empty.
SMOKE-labelled bundles contain synthetic checks only and must not be used as research results.
''')
    partial = bool(failures or (plan.get('benchmark_requested') and partial_profile_points)
                   or (plan.get('dynamic_requested') and dynamic_missing))
    coverage['requested_measurements_complete'] = not partial
    (output / 'coverage.json').write_text(json.dumps(coverage, indent=2))
    archive_name = ('SMOKE_NOT_PAPER_RESULTS' if plan['synthetic'] else 'PAPER_RESULTS') + ('_PARTIAL' if partial else '') + '.zip'
    archive = suite / archive_name
    files = [(p, str(p.relative_to(output))) for p in output.rglob('*') if p.is_file() and '.matplotlib_cache' not in p.parts]
    files += raw_files + [(suite / 'suite_plan.json', 'suite_plan.json')]
    if (suite / 'suite_status.json').exists(): files.append((suite / 'suite_status.json', 'suite_status.json'))
    files += [(p, 'resolved_configs/' + p.name) for p in (suite / 'resolved_configs').glob('*.json')]
    files += [(p, 'source_code/' + p.name) for p in project.iterdir() if p.suffix in ('.py', '.md', '.ipynb', '.txt')]
    hashes = {}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
        for path, name in files:
            if name in hashes: continue
            payload = path.read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            handle.writestr(name, payload)
        handle.writestr('FILE_HASHES.json', json.dumps(hashes, indent=2))
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None: raise ValueError('Results ZIP integrity failed')
    print(f'Results bundle: {archive}', flush=True)
    print(f'Completed {len(inventory)}/{coverage["expected_method_seed_runs"]} runs; synthetic={plan["synthetic"]}', flush=True)
    return archive


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--include-checkpoints', action='store_true')
    p.add_argument('--no-figures', action='store_true')
    args = p.parse_args()
    export(args.suite, args.include_checkpoints, args.no_figures)
