import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from export_results import aggregate, export, csv_write
from run_suite import make_plan, job_folder, METHODS


class ResultsTests(unittest.TestCase):
    def test_revised_main_and_distillation_ablation(self):
        project = Path(__file__).parent
        for dataset in ('fashionmnist', 'cifar10', 'cifar100'):
            plan = make_plan([dataset], ['main', 'ablations', 'sensitivity'], [42], 'train', project)
            for job in plan['jobs']:
                expected = .2 if job['id'].endswith('/with_distillation') else 0.
                self.assertEqual(job['config']['kd_alpha'], expected)
            self.assertEqual(sum(j['id'].endswith('/with_distillation') for j in plan['jobs']), 1)
            self.assertFalse(any(j['id'].endswith('/no_distillation') for j in plan['jobs']))

    def test_suite_covers_all_requested_groups(self):
        plan = make_plan(['fashionmnist'], ['main', 'ablations', 'sensitivity'], [42, 123, 2024], 'train', Path(__file__).parent)
        self.assertEqual(len(plan['jobs']), 19)
        self.assertEqual({j['method'] for j in plan['jobs'] if j['category'] == 'main'}, set(METHODS))
        self.assertEqual(len([j for j in plan['jobs'] if j['category'] == 'ablations']), 4)
        self.assertEqual(len([j for j in plan['jobs'] if j['category'] == 'sensitivity']), 8)
        self.assertTrue(all(j['config']['training_clients'] == 20 for j in plan['jobs']))
        for job in plan['jobs']:
            if job['category'] == 'sensitivity': self.assertEqual(len(job['overrides']), 1)

    def test_seed_aggregation_and_no_fake_zero_sd(self):
        rows = [{'method': 'x', 'seed': s, 'acc': a, 'synthetic': False} for s, a in [(42, .4), (123, .6), (2024, .8)]]
        result = aggregate(rows, ['method'], ['acc'])[0]
        self.assertAlmostEqual(result['acc_mean'], .6)
        self.assertAlmostEqual(result['acc_std'], .2)
        self.assertIsNone(aggregate(rows[:1], ['method'], ['acc'])[0]['acc_std'])
        with self.assertRaises(ValueError): aggregate(rows + rows[:1], ['method'], ['acc'])

    def test_portable_bundle_and_missing_measurement_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = make_plan(['fashionmnist'], ['main'], [42, 123, 2024], 'train', Path(__file__).parent)
            plan['jobs'] = [j for j in plan['jobs'] if j['method'] == 'fedavg']
            plan.update(benchmark_requested=True, dynamic_requested=True)
            (root / 'suite_plan.json').write_text(json.dumps(plan))
            job = plan['jobs'][0]
            for seed, acc in [(42, .4), (123, .6), (2024, .8)]:
                folder = job_folder(root, job, seed, 'train')
                folder.mkdir(parents=True)
                (folder / 'manifest.json').write_text(json.dumps({'evaluation_complete': True, 'synthetic_smoke': False,
                    'config': job['config'], 'seed': seed, 'method': 'fedavg', 'split_sha256': f'split-{seed}',
                    'deployment_checkpoint_sha256': 'fixture-checkpoint'}))
                (folder.parent / 'splits.json').write_text('{}')
                (folder / 'final_validation.json').write_text('{}')
                with gzip.open(folder / 'predictions.jsonl.gz', 'wt') as handle: handle.write('')
                csv_write(folder / 'clients.csv', [{'variant': 'global', 'exit': 'final', 'requested_removal': 0,
                    'client': c, 'acc': acc, 'macro_f1': acc / 2, 'balanced_acc': acc, 'loss': 1.0,
                    'compact_path_parameters': 100, 'active_output_structures': 10, 'total_output_structures': 10,
                    'active_output_structure_ratio': 1, 'achieved_group_removal': 0, 'compact_logit_max_error': 0}
                    for c in range(20)])
                (folder / 'history.json').write_text(json.dumps([{'round': 1, 'selection_score': .2,
                    'validation_accuracy': {'0': .2}, 'validation_metrics': {'0': {'loss': 1.}},
                    'round_wall_seconds': 1., 'mean_objective_over_student_passes': 1.,
                    'retained_set_jaccard_at_40_previous_round': 1., 'student_forward_passes': 20, 'optimizer_steps': 20}]))
            path = export(root, no_figures=True)
            self.assertEqual(path.name, 'PAPER_RESULTS_PARTIAL.zip')
            coverage = json.loads((root / 'paper_results/coverage.json').read_text())
            self.assertFalse(coverage['requested_measurements_complete'])
            self.assertFalse(coverage['gpu_memory_measurements_available'])
            with (root / 'paper_results/tables/table_01_overall.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[0]['acc_mean']), .6)
            self.assertAlmostEqual(float(rows[0]['acc_std']), .2)
            with zipfile.ZipFile(path) as archive:
                hashes = json.loads(archive.read('FILE_HASHES.json'))
                for name, sha in hashes.items(): self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), sha)
                self.assertIn('raw/main/fashionmnist/fedavg/seed_42/predictions.jsonl.gz', archive.namelist())


if __name__ == '__main__':
    unittest.main()
