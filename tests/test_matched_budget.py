import unittest
from matched_budget import select
class SelectionTests(unittest.TestCase):
    def test_no_test_leakage(self):
        rows=[dict(parameters=8,validation_accuracy=.8,test_accuracy=.1,level=20),dict(parameters=7,validation_accuracy=.7,test_accuracy=.99,level=30)]
        self.assertEqual(select(rows,8)['level'],20)
        self.assertIsNone(select(rows,6))
    def test_budget_and_ties(self):
        rows=[dict(parameters=9,validation_accuracy=.99,level=10),dict(parameters=8,validation_accuracy=.8,level=20),dict(parameters=7,validation_accuracy=.8,level=30)]
        self.assertEqual(select(rows,8)['parameters'],7)
if __name__=='__main__':unittest.main()
