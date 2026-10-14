"""Tests for the independent native nested-shape probability audit math."""
import unittest
import math

import numpy as np

from scripts import audit_nested_shapes as audit


class NestedProbabilityAuditTests(unittest.TestCase):
    def test_integer_beta_cdf_matches_independent_lower_binomial_tail(self):
        t = np.asarray([0., .01, .1, .3, .5, .77, .99, 1.], dtype=np.float64)
        for degree in (1, 3, 6, 12):
            got = audit._beta_basis_cdf(t, degree)
            order = degree + 1
            expected = np.empty_like(got)
            for k in range(degree + 1):
                lower_tail = np.zeros_like(t)
                for j in range(k + 1):
                    lower_tail += (math.comb(order, j) * t ** j * (1. - t) ** (order - j))
                expected[:, k] = 1. - lower_tail
            np.testing.assert_allclose(got, expected, atol=2e-15, rtol=2e-14)
            self.assertTrue(np.all((got >= -1e-14) & (got <= 1. + 1e-14)))
            self.assertTrue(np.all(np.diff(got, axis=0) >= -1e-14))

    def test_registered_reference_shapes_are_censored_probability_laws(self):
        anchors = np.asarray([-1., -.63, 0., .41, 1.], dtype=np.float32)
        grid = np.broadcast_to(np.linspace(-1.05, 1.05, 401)[None, :], (len(anchors), 401))
        for case in audit.CASES:
            raw = np.zeros((len(anchors), audit._expected_output_dim(case)), dtype=np.float64)
            params = audit.reference_parameters(case, anchors, raw)
            cdf = audit.reference_cdf(params, grid)
            bins = audit.reference_bins(params)
            quantiles = audit.reference_quantiles(params)
            self.assertTrue(np.all(np.diff(cdf, axis=1) >= -2e-13), case)
            self.assertTrue(np.all((cdf >= 0.) & (cdf <= 1.)), case)
            np.testing.assert_allclose(bins.sum(axis=1), 1., atol=2e-14, rtol=0.)
            self.assertTrue(np.all(bins >= -2e-14), case)
            if case != "mdn":
                np.testing.assert_array_equal(quantiles[:, 8], anchors.astype(np.float64))
                np.testing.assert_allclose(audit.reference_cdf(params, anchors)[:, 0][1:-1], .5,
                                           atol=3e-13, rtol=0., err_msg=case)
            self.assertEqual(float(audit.reference_cdf(params, np.full((len(anchors), 1), -1.1))[0, 0]), 0.)
            self.assertEqual(float(audit.reference_cdf(params, np.full((len(anchors), 1), 1.1))[0, 0]), 1.)

    def test_median_mdn_implicit_gradient_matches_independent_recentered_cdf(self):
        import torch
        from jevaffect import shape_controls

        anchor_value = .23
        query = .61
        anchor = torch.tensor([anchor_value], dtype=torch.float64)
        raw_scale = torch.tensor([-.9], dtype=torch.float64, requires_grad=True)
        logits = torch.linspace(-.5, .8, 15, dtype=torch.float64)[None].requires_grad_()
        output = shape_controls.mdn_cdf(
            shape_controls.mdn_parameters(anchor, raw_scale, logits, recenter=True),
            torch.tensor([query], dtype=torch.float64),
        ).sum()
        output.backward()
        self.assertIsNone(anchor.grad)

        def independent(raw_s, raw_logits):
            scale = np.logaddexp(0., raw_s) + 1e-4
            z = raw_logits - np.max(raw_logits)
            weights = np.exp(z) / np.exp(z).sum()
            centers = np.linspace(-1.4, 1.4, 15)
            lo, hi = centers[0] - 40 * scale, centers[-1] + 40 * scale
            for _ in range(100):
                mid = (lo + hi) / 2
                cdf = np.sum(weights / (1. + np.exp(-np.clip((mid - centers) / scale, -700, 700))))
                if cdf < .5:
                    lo = mid
                else:
                    hi = mid
            median = (lo + hi) / 2
            shift = anchor_value - median
            x = (query - shift - centers) / scale
            return float(np.sum(weights / (1. + np.exp(-np.clip(x, -700, 700)))))

        eps = 1e-5
        base_scale = float(raw_scale.detach())
        base_logits = logits.detach().numpy()[0]
        finite_scale = (independent(base_scale + eps, base_logits)
                        - independent(base_scale - eps, base_logits)) / (2 * eps)
        self.assertAlmostEqual(float(raw_scale.grad[0]), finite_scale, delta=2e-7)
        for k in (0, 7, 14):
            plus, minus = base_logits.copy(), base_logits.copy()
            plus[k] += eps
            minus[k] -= eps
            finite = (independent(base_scale, plus) - independent(base_scale, minus)) / (2 * eps)
            self.assertAlmostEqual(float(logits.grad[0, k]), finite, delta=2e-7)

    def test_nested_source_roles_require_partition_and_once_only_outer_ids(self):
        ids = [f"source-{i:02d}/clip-00" for i in range(15)]
        roles = ("point_fit", "point_select", "shape_fit", "shape_select", "outer")
        folds = []
        for fold in range(3):
            groups = {name: [] for name in roles}
            for i, sid in enumerate(ids):
                source_index = int(audit.source_of(sid).split("-")[1])
                if fold * 5 <= source_index < (fold + 1) * 5:
                    groups["outer"].append(sid)
                else:
                    role_index = (source_index + fold) % 4
                    groups[roles[role_index]].append(sid)
            holdout = groups["shape_fit"] + groups["shape_select"] + groups["outer"]
            folds.append({"fold": fold, **{name + "_ids": groups[name] for name in roles},
                          "outer_ids": groups["outer"], "point_oof_ids": holdout, "oof_ids": holdout,
                          "fit_ids": groups["point_fit"], "select_ids": groups["point_select"],
                          "point_scaler_fit_ids": groups["point_fit"]})
        audit._assert_roles({"folds": folds}, ids)
        folds[1]["point_select_ids"].append(folds[1]["point_fit_ids"][0])
        with self.assertRaisesRegex(audit.AuditError, "duplicate|partition|share source"):
            audit._assert_roles({"folds": folds}, ids)

    def test_raw_point_must_match_across_cases_within_seed_but_may_change_between_seeds(self):
        def records(point):
            return [{"id": "source/clip", "native_point": point, "target": .2, "anchor": .1}]

        merged = {
            audit.PRIMARY: {13: records(.1), 42: records(.2)},
            "scale": {13: records(.1), 42: records(.2)},
        }
        self.assertNotEqual(merged[audit.PRIMARY][13][0]["native_point"],
                            merged[audit.PRIMARY][42][0]["native_point"])
        self.assertTrue(audit.same_seed_raw_point_invariant(
            merged, cases=(audit.PRIMARY, "scale"), seeds=(13, 42)))
        merged["scale"][42] = records(.1)
        self.assertFalse(audit.same_seed_raw_point_invariant(
            merged, cases=(audit.PRIMARY, "scale"), seeds=(13, 42)))

    def test_hit_event_uses_unclipped_raw_point_and_float32_tolerance(self):
        anchor = np.asarray([-1., .4, 1.], dtype=np.float32)
        raw = np.zeros((3, audit._expected_output_dim("scale")), dtype=np.float64)
        params = audit.reference_parameters("scale", anchor, raw)
        raw_point = np.asarray([-1.05, .4, 1.05], dtype=np.float32)
        probability = audit.reference_hit_probability(params, raw_point)
        self.assertEqual(probability.shape, (3,))
        self.assertTrue(np.all(probability >= 0.))
        self.assertTrue(np.all(probability <= 1.))
        # The endpoint anchor contributes the censoring atom when the raw
        # point's event interval touches the native boundary.
        self.assertGreater(probability[0], 0.)
        self.assertGreater(probability[2], 0.)


if __name__ == "__main__":
    unittest.main()
