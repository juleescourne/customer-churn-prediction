"""Regression checks for raw-data evaluation, not model-performance claims."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scripts.evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'data.csv'
        self.output = Path(self.tmp.name) / 'report.json'
        rng = np.random.default_rng(27)
        n = 240
        self.data = pd.DataFrame({
            'CustomerId': np.arange(n), 'CreditScore': rng.integers(400, 800, n),
            'Geography': rng.choice(['France', 'Germany'], n),
            'Gender': rng.choice(['Female', 'Male'], n), 'Age': rng.integers(20, 70, n),
            'Tenure': rng.integers(0, 10, n), 'Balance': rng.uniform(0, 100000, n),
            'NumOfProducts': rng.integers(1, 4, n), 'HasCrCard': rng.integers(0, 2, n),
            'IsActiveMember': rng.integers(0, 2, n),
            'EstimatedSalary': rng.uniform(10000, 80000, n), 'Exited': np.tile([0, 0, 0, 1], 60)})

    def run_evaluation(self):
        self.data.to_csv(self.source, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate(self.source, self.output)
        return json.loads(self.output.read_text())

    def test_excluded_target_proxy_cannot_change_predictions(self):
        self.data['Complain'] = self.data.Exited
        self.data['Surname'] = 'historical-name'
        first = self.run_evaluation()
        self.data['Complain'] = 1 - self.data.Exited
        self.data['Surname'] = 'different-name'
        second = self.run_evaluation()
        self.assertEqual(first['test'], second['test'])
        self.assertEqual(first['validation_average_precision'], second['validation_average_precision'])
        self.assertNotIn('Complain', second['features'])
        self.assertNotEqual(first['source_sha256'], second['source_sha256'])
        self.assertEqual(sum(first['split'][k] for k in ['train', 'validation', 'test']), len(self.data))
        for score in first['test'].values():
            self.assertEqual(sum(map(sum, score['confusion_matrix'])), first['split']['test'])

    def test_duplicate_customer_is_rejected_before_output(self):
        self.data.loc[1, 'CustomerId'] = self.data.loc[0, 'CustomerId']
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.run_evaluation()
        self.assertFalse(self.output.exists())

    def test_single_class_is_rejected(self):
        self.data['Exited'] = 0
        with self.assertRaisesRegex(ValueError, 'Both target classes'):
            self.run_evaluation()

    def test_missing_target_is_rejected(self):
        self.data['Exited'] = self.data.Exited.astype(float)
        self.data.loc[0, 'Exited'] = np.nan
        with self.assertRaisesRegex(ValueError, 'binary and complete'):
            self.run_evaluation()
