"""Exercise the real-data CLI path with torchvision-style target storage."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import run


class DatasetLabelTests(unittest.TestCase):
    def arguments(self):
        return ['run.py', '--config', str(Path(__file__).parent / 'configs/fashionmnist.json'),
                '--method', 'fedavg', '--seeds', '42', '--rounds', '1', '--device', 'cpu']

    def test_real_data_path_normalizes_tensor_numpy_and_list_targets(self):
        for targets in (torch.arange(10).repeat(4), np.tile(np.arange(10), 4), list(range(10)) * 4):
            with self.subTest(storage=type(targets).__name__):
                train = SimpleNamespace(targets=targets)
                heldout = SimpleNamespace(targets=torch.arange(10).repeat(2))
                with patch('sys.argv', self.arguments()), patch('run.load_data', return_value=(train, train, heldout, {})), \
                     patch('run.shared_splits', return_value={}) as split, patch('run.run_one') as train_run:
                    run.main()
                self.assertEqual(train_run.call_count, 1)
                train_labels, test_labels = split.call_args.args[2:4]
                self.assertEqual(train_labels, list(range(10)) * 4)
                self.assertEqual(test_labels, list(range(10)) * 2)
                self.assertTrue(all(type(v) is int for v in train_labels + test_labels))

    def test_real_data_path_still_rejects_missing_or_invalid_classes(self):
        for targets in (torch.arange(9), torch.arange(11)):
            dataset = SimpleNamespace(targets=targets)
            with patch('sys.argv', self.arguments()), patch('run.load_data', return_value=(dataset, dataset, dataset, {})), \
                 patch('run.shared_splits') as split, patch('run.run_one') as train_run:
                with self.assertRaisesRegex(ValueError, 'missing=.*unexpected='):
                    run.main()
                split.assert_not_called()
                train_run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
