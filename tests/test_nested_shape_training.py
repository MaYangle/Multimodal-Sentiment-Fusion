import copy
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from tests import test_nested_shape_points as point_fixtures
from jevaffect import nested_shape_training as st
from jevaffect.native_training import digest


class NestedShapeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.points = point_fixtures.NestedShapePointTests()
        self.points.setUp()
        self.root = self.points.root
        from jevaffect.nested_shape_points import run_nested_suite
        self.suite = self.root/'point_suite.json'
        run_nested_suite(self.points.config, output=self.suite, allow_fixture=True)
        self.tables, self.binding = st.load_stage(self.suite, 13, 0, allow_smoke=True)
        self.config = dict(run_root=str(self.root/'shapes'), device='cpu', torch_threads=1,
                           hidden=8, dropout=.1, learning_rate=.001, weight_decay=.0001,
                           batch_size=4, max_epochs=3, patience=12, gradient_norm=1.)

    def tearDown(self):
        self.points.tearDown()

    def test_all_case_real_backward_and_checkpoint_reload(self):
        for case in st.shapes.CASES:
            original_diagnostics = st.diagnostics
            def diagnose(anchor, target, output):
                try:
                    return original_diagnostics(anchor, target, output)
                except ValueError as error:
                    bins = np.asarray(output['bins'], np.float64)
                    detail = dict(case=case, shape=bins.shape,
                                  minimum_probability=float(bins.min()),
                                  maximum_row_mass_error=float(np.abs(bins.sum(1)-1).max()),
                                  finite=bool(np.isfinite(bins).all()))
                    raise ValueError(f"{error}; numerical fixture detail: {detail}") from error
            with patch.object(st, 'diagnostics', side_effect=diagnose):
                result = st.train(self.config, self.tables, self.binding, case=case, seed=13, fold=0)
            self.assertTrue(result['complete'])
            self.assertEqual(result['epochs'], 3)
            self.assertGreater(result['updates'], 0)
            records = json.loads((Path(result['root'])/'outer_predictions.json').read_text())
            self.assertEqual([r['id'] for r in records], self.tables['outer']['ids'])
            self.assertTrue(all(np.isfinite(r['CRPS_trapezoid_1001']) for r in records))
            resumed = st.train(self.config, self.tables, self.binding, case=case, seed=13, fold=0, resume=True)
            self.assertEqual(result['predictions_sha256'], resumed['predictions_sha256'])

    def test_resume_is_exact_and_outer_labels_cannot_select_or_fit(self):
        case = 'median_mdn'
        whole = st.train(self.config, self.tables, self.binding, case=case, seed=42, fold=1)
        cfg = {**self.config, 'run_root': str(self.root/'partial')}
        partial = st.train(cfg, self.tables, self.binding, case=case, seed=42, fold=1, stop_after=1)
        self.assertFalse(partial['complete'])
        self.assertFalse((Path(partial['root'])/'outer_predictions.json').exists())
        resumed = st.train(cfg, self.tables, self.binding, case=case, seed=42, fold=1, resume=True)
        changed = copy.deepcopy(self.tables)
        changed['outer']['target'] *= -.4
        cfg2 = {**self.config, 'run_root': str(self.root/'changed')}
        outside = st.train(cfg2, changed, self.binding, case=case, seed=42, fold=1)
        states = [torch.load(Path(r['root'])/'last.pt', weights_only=False) for r in (whole, resumed, outside)]
        for state in states[1:]:
            for key, value in states[0]['model'].items():
                self.assertTrue(torch.equal(value, state['model'][key]), key)
            for a, b in zip(states[0]['history'], state['history']):
                self.assertEqual({k:v for k,v in a.items() if k != 'seconds'}, {k:v for k,v in b.items() if k != 'seconds'})
        self.assertEqual(whole['best_epoch'], outside['best_epoch'])

    def test_table_tampering_and_unmarked_smoke_rejected(self):
        with self.assertRaises(ValueError):
            st.load_stage(self.suite, 13, 0)
        suite = json.loads(self.suite.read_text())
        path = Path(suite['results'][0]['tables']['shape_fit']['path'])
        content = path.read_bytes(); path.write_bytes(content+b'changed')
        with self.assertRaisesRegex(ValueError, 'table changed'):
            st.load_stage(self.suite, 13, 0, allow_smoke=True)
        path.write_bytes(content)


if __name__ == '__main__':
    unittest.main()
