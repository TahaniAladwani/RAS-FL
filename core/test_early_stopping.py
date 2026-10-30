import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset
from early_stopping import EarlyStopping
from run import run_one


class EarlyStoppingTests(unittest.TestCase):
    def test_three_ties_stop_after_baseline(self):
        s=EarlyStopping(3)
        self.assertEqual([s.update(.5) for _ in range(4)], [False,False,False,True])

    def test_improvement_resets_and_declines_count(self):
        s=EarlyStopping(3)
        self.assertEqual([s.update(x) for x in [.5,.4,.4,.6,.5,.4,.3]], [False]*6+[True])

    def test_min_delta_and_resume_state(self):
        s=EarlyStopping(3,.01)
        for x in [.5,.505,.5]:s.update(x)
        restored=EarlyStopping(**s.state_dict())
        self.assertTrue(restored.update(.49))
        self.assertTrue(restored.update(.9))  # A stopped run stays stopped on resume.
        with self.assertRaises(ValueError):EarlyStopping().update(float('nan'))

    def test_training_loop_stops_restores_best_and_resumes_without_extra_round(self):
        cfg=json.loads((Path(__file__).parent/'configs/fashionmnist.json').read_text())
        cfg.update(rounds=10,early_stopping_patience=3)
        data=TensorDataset(torch.randn(80,1,28,28),torch.arange(80)%10)
        split={'train_clients':[{'train':list(range(4*c,4*c+3)), 'validation':[4*c+3]} for c in range(20)],
               'deployment_clients':[{'adapt':[2*c],'test':[2*c+1]} for c in range(20)]}
        cost={'student_forward_passes':1,'optimizer_steps':1,'objective_loss_sum_over_student_passes':1.,'objective_student_pass_count':1}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'
            with patch('run.local_train',return_value=cost) as train, patch('run.validation_score',return_value=(.5,{0:.5})), patch('run.evaluate_deployment',return_value=[{'acc':.5}]):
                run_one(cfg,'fedavg',42,data,data,data,split,folder,torch.device('cpu'),True)
                self.assertEqual(train.call_count,80)  # 4 rounds, 20 clients each.
                stop=json.loads((folder/'training_stop.json').read_text())
                self.assertEqual(stop['completed_rounds'],4)
                self.assertEqual(stop['reason'],'early_stopping')
                saved=torch.load(folder/'deployment.pt',weights_only=False)
                self.assertEqual(saved['selected_round'],1)
                m=json.loads((folder/'manifest.json').read_text());m.pop('evaluation_complete')
                (folder/'manifest.json').write_text(json.dumps(m))
                run_one(cfg,'fedavg',42,data,data,data,split,folder,torch.device('cpu'),True,resume=True)
                self.assertEqual(train.call_count,80)
                self.assertEqual(json.loads((folder/'training_stop.json').read_text())['completed_rounds'],4)


if __name__=='__main__':unittest.main()
