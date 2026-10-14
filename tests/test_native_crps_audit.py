import unittest

import numpy as np
import torch

from scripts import audit_native_crps as audit
from scripts import audit_nested_shapes as reference
from jevaffect import cdf_crps, shape_controls


class NativeCrpsAuditMathTests(unittest.TestCase):
    def test_split_quadrature_matches_uniform_and_atomic_analytic_limits(self):
        anchors = np.asarray([-1., -.63, .17, .81, 1.], dtype=np.float32)
        targets = np.asarray([.312345, -.1, .53, -1., .987654], dtype=np.float32)

        def uniform_cdf(grid):
            return np.clip((grid + 1.) / 2., 0., 1.)

        expected_uniform = ((targets.astype(np.float64) + 1.) ** 3
                            + (1. - targets.astype(np.float64)) ** 3) / 12.
        got_uniform = audit._split_gauss_crps(uniform_cdf, anchors, targets, 48)
        np.testing.assert_allclose(got_uniform, expected_uniform, rtol=0., atol=2e-14)

        atoms = np.asarray([-1., -.35, .57, .83, 1.], dtype=np.float64)

        def atomic_cdf(grid):
            return (grid >= atoms[:, None]).astype(np.float64)

        got_atomic = audit._split_gauss_crps(atomic_cdf, atoms, targets, 48)
        np.testing.assert_allclose(got_atomic, np.abs(atoms - targets), rtol=0., atol=2e-14)

    def test_registered_numpy_reconstruction_is_finite_bounded_and_node_stable(self):
        anchors = np.asarray([-1., -.73, -.1, .314159, .82, 1.], dtype=np.float32)
        targets = np.asarray([-.876543, -.299999, .123456, -.912345, .777777, .999999], dtype=np.float32)
        raw_by_case = {
            "scale": np.asarray([[-2.1], [-.3], [.2], [1.1], [-1.2], [.4]], dtype=np.float64),
            "mixture": np.asarray([[-1.7, -.8], [-.2, .1], [.5, -1.1], [1.2, .7], [-.8, -2.], [.3, .4]]),
            "median_mdn": np.tile(np.linspace(-.7, .8, 16), (len(anchors), 1)),
            "bern_d6": np.tile(np.linspace(-.4, .7, 16), (len(anchors), 1)),
        }
        for case, raw in raw_by_case.items():
            with self.subTest(case=case):
                params = reference.reference_parameters(case, anchors, raw)
                bins = reference.reference_bins(params)
                quantiles = reference.reference_quantiles(params)
                self.assertTrue(np.isfinite(bins).all())
                self.assertTrue(np.isfinite(quantiles).all())
                self.assertTrue(np.all(quantiles >= -1.))
                self.assertTrue(np.all(quantiles <= 1.))
                self.assertTrue(all(np.all(quantiles[:, j] <= quantiles[:, j + 1])
                                    for j in range(0, 8, 2)))
                if case != "mdn":
                    np.testing.assert_allclose(quantiles[:, 8], anchors, rtol=0., atol=1e-12)
                scores = {nodes: audit.reference_crps(params, targets, nodes)
                          for nodes in audit.QUADRATURE_NODES}
                self.assertTrue(all(np.isfinite(score).all() for score in scores.values()))
                self.assertLessEqual(float(np.abs(scores[48] - scores[192]).mean()), 1e-4)

    def test_numpy_replay_matches_cpu_cdf_bins_events_quantiles_and_training_crps(self):
        torch.manual_seed(9021)
        anchors = np.asarray([-1., -.62, -.17, .24, .71, 1.], dtype=np.float32)
        targets = np.asarray([-.41, .123456, -.999999, .876543, .12, -.2], dtype=np.float32)
        context = torch.randn(len(anchors), 7, dtype=torch.float32)
        query = np.broadcast_to(np.linspace(-1., 1., 23, dtype=np.float64),
                                (len(anchors), 23)).copy()
        boundaries = reference.BOUNDARIES.astype(np.float64)
        for case in audit.CASES:
            with self.subTest(case=case):
                model = shape_controls.ShapeHead(case, feature_dim=7, hidden=6, dropout=0.).eval()
                anchor_tensor = torch.from_numpy(anchors)
                target_tensor = torch.from_numpy(targets)
                output = model(context, anchor_tensor)
                raw = reference._linear_head_outputs(model.state_dict(), context.numpy())
                params = reference.reference_parameters(case, anchors, raw)

                torch_cdf = shape_controls.cdf(
                    output, torch.from_numpy(query)).detach().cpu().numpy()
                np.testing.assert_allclose(torch_cdf, reference.reference_cdf(params, query),
                                           rtol=0., atol=2e-6)
                torch_bins = shape_controls.probabilities(output, boundaries).detach().cpu().numpy()
                np_bins, _ = audit._sanitize_reference_bins(
                    params, negative_tolerance=1e-12, row_sum_tolerance=1e-10,
                    cdf_range_tolerance=1e-12)
                np.testing.assert_allclose(torch_bins, np_bins, rtol=0., atol=2e-6)
                torch_hit = shape_controls.hit_probability(
                    output, anchor_tensor, float(np.float32(.1))).detach().cpu().numpy()
                np_hit = audit.reference_native_hit_probability(params, anchors)
                np.testing.assert_allclose(torch_hit, np_hit, rtol=0., atol=2e-6)
                levels = [.25, .75, .1, .9, .05, .95, .025, .975, .5]
                torch_quantiles = shape_controls.quantiles(output, levels).cpu().numpy()
                np.testing.assert_allclose(torch_quantiles, reference.reference_quantiles(params),
                                           rtol=0., atol=2e-6)
                torch_score = cdf_crps.per_sample(output, target_tensor, nodes=48).detach().cpu().numpy()
                np_score = audit.reference_crps(params, targets, nodes=48)
                np.testing.assert_allclose(torch_score, np_score, rtol=0., atol=2e-6)

    def test_native_event_mass_includes_censoring_atoms_and_keeps_raw_center_auxiliary(self):
        anchors = np.asarray([-1., 1.], dtype=np.float32)
        raw = np.asarray([[.4], [.4]], dtype=np.float64)
        params = reference.reference_parameters("scale", anchors, raw)
        probability = audit.reference_native_hit_probability(params, anchors)
        low_endpoint = reference.reference_cdf(params, np.asarray([-.9, .9]))[:, 0]
        expected = np.asarray([low_endpoint[0], 1. - low_endpoint[1]])
        np.testing.assert_allclose(probability, expected, rtol=0., atol=1e-9)
        # A raw point beyond the native bound has a different event center;
        # the audit retains it as a separately named auxiliary prediction.
        raw_probability = audit.reference_native_hit_probability(params, np.asarray([-1.2, 1.2], np.float32))
        self.assertFalse(np.array_equal(raw_probability, probability))

    def test_off_grid_targets_are_accepted_and_out_of_bound_targets_rejected(self):
        anchor = np.asarray([-.2, .4], dtype=np.float32)
        off_grid = np.asarray([.123456, -.876543], dtype=np.float32)
        cdf = lambda grid: np.clip((grid + 1.) / 2., 0., 1.)
        result = audit._split_gauss_crps(cdf, anchor, off_grid, 48)
        self.assertTrue(np.isfinite(result).all())
        with self.assertRaises(ValueError):
            audit._split_gauss_crps(cdf, anchor, np.asarray([0., 1.0001]), 48)
        with self.assertRaises(ValueError):
            audit._split_gauss_crps(cdf, np.asarray([0., 1.01]), np.asarray([0., .2]), 48)
        for bad_nodes in (7, True, 49, 1.5):
            with self.subTest(nodes=bad_nodes), self.assertRaises(ValueError):
                audit._split_gauss_crps(cdf, anchor, off_grid, bad_nodes)

    def test_only_declared_bin_roundoff_is_clamped_and_renormalized(self):
        raw = np.asarray([[.2, .3, -5e-13, .1, .4000000000005]], dtype=np.float64)
        repaired, metadata = audit._repair_reference_bins(
            raw, negative_tolerance=1e-12, row_sum_tolerance=1e-10)
        self.assertEqual(metadata["repaired_rows"], 1)
        self.assertTrue(np.all(repaired >= 0.))
        np.testing.assert_allclose(repaired.sum(axis=1), 1., rtol=0., atol=1e-15)
        meaningful_negative = np.asarray([[.2, .3, -2e-12, .1, .400000000002]], dtype=np.float64)
        with self.assertRaises(audit.AuditError):
            audit._repair_reference_bins(meaningful_negative, negative_tolerance=1e-12,
                                         row_sum_tolerance=1e-10)

    def test_point_teacher_invariance_is_checked_within_seed_only(self):
        merged = {case: {} for case in audit.CASES}
        for case in audit.CASES:
            for seed in audit.SEEDS:
                # Different seeds deliberately use different point teachers.
                point = float(seed) / 10000.
                merged[case][seed] = [{"id": "clip/source", "native_point": point,
                                       "target": .25, "anchor": point}]
        self.assertTrue(audit._same_seed_frozen_point(merged))
        merged["mixture"][13][0]["native_point"] += 1e-3
        self.assertFalse(audit._same_seed_frozen_point(merged))


if __name__ == "__main__":
    unittest.main()
