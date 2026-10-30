import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce


class ReleaseTests(unittest.TestCase):
    def test_complete_grid(self):
        args = reproduce.parser().parse_args([])
        tasks = reproduce.commands(args)
        self.assertEqual(sum(t['runs'] for t in tasks), 450)
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({t['name'] for t in tasks}), 8)
        for alpha in ['0.1', '0.4']:
            self.assertEqual(sum(t['runs'] for t in tasks if t['name'].startswith('alpha_' + alpha + '/')), 225)

    def test_shared_patience_defaults_and_override(self):
        for value in [5, 7]:
            args = reproduce.parser().parse_args([] if value == 5 else ['--patience', str(value)])
            for task in reproduce.commands(args):
                command = task['command']
                flag = '--early-stopping-patience' if task['name'].endswith('/core') else '--patience'
                self.assertEqual(command[command.index(flag) + 1], str(value))
                self.assertNotIn('--rasfl-early-stopping-patience', command)
        for path in list((reproduce.ROOT/'core/configs').glob('*.json')) + list((reproduce.ROOT/'baselines').glob('*/base_config.json')):
            cfg = json.loads(path.read_text())
            self.assertEqual(cfg['early_stopping_patience'], 5)
            self.assertNotIn('early_stopping_patience_rasfl', cfg)

    def test_level_sets_and_checkpoint_selection(self):
        plan = reproduce.suite.make_plan(reproduce.DATASETS, ['ablations', 'levels'], [42,123,2024], 'train', reproduce.ROOT/'core')
        for dataset in reproduce.DATASETS:
            jobs = [j for j in plan['jobs'] if j['dataset'] == dataset]
            levels = [j['config']['training_removal_levels'] for j in jobs if j['category'] == 'levels']
            self.assertEqual(levels, [[10,40], [10,25,40], [5,10,20,30,40], [5,10,15,20,25,30,40]])
            selection = next(j for j in jobs if j['id'].endswith('full_model_selection'))
            self.assertEqual(selection['config']['full_validation_weight'], 1.)

    def test_sensitivity_reference_is_included(self):
        plan = reproduce.suite.make_plan(reproduce.DATASETS, ['sensitivity'], [42], 'train', reproduce.ROOT/'core')
        for dataset in reproduce.DATASETS:
            self.assertEqual(sum(j['id'] == f'main/{dataset}/rasfl' for j in plan['jobs']), 1)

    def test_baseline_dataset_configuration(self):
        for dataset, classes in [('fashionmnist',10),('cifar10',10),('cifar100',100)]:
            cfg=json.loads((reproduce.ROOT/'baselines'/dataset/'base_config.json').read_text())
            self.assertEqual(cfg['dataset'], dataset)
            self.assertEqual(cfg['classes'], classes)


if __name__ == '__main__':
    unittest.main()
