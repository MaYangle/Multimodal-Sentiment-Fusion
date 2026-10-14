"""Small deterministic checks for the independent nested point-stage audit."""
import unittest

import numpy as np

from scripts import audit_nested_shape_points as audit


class NestedPointAuditTests(unittest.TestCase):
    def test_source_roles_replay_as_complete_disjoint_partitions(self):
        ids = [f"source-{source:02d}/clip-{clip:02d}"
               for source in range(12) for clip in range(2)]
        # Reverse CSV order so role output order must follow the supplied
        # canonical cohort rather than sorted source order or cache order.
        ids.reverse()
        outer, rebuilt = audit.rebuild_source_roles(ids, expected_source_count=12)
        self.assertEqual(len(outer), 3)
        self.assertEqual(set().union(*(set(group) for group in outer)), set(ids))
        self.assertEqual(sum(map(len, outer)), len(ids))
        for fold in rebuilt["folds"]:
            roles = fold["role_ids"]
            self.assertEqual(set().union(*(set(roles[name]) for name in audit.ROLES)), set(ids))
            for left_index, left in enumerate(audit.ROLES):
                for right in audit.ROLES[left_index + 1:]:
                    self.assertFalse({audit.source_of(sid) for sid in roles[left]}
                                     & {audit.source_of(sid) for sid in roles[right]})
            self.assertEqual(
                set(fold["point_oof_ids"]),
                set(roles["shape_fit"]) | set(roles["shape_select"]) | set(roles["outer"]),
            )
            self.assertEqual(roles["outer"], [sid for sid in ids if sid in set(roles["outer"])])

    def test_sequence_pooling_respects_lengths_and_text_empty_rows(self):
        text = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        lengths = np.array([3, 0], dtype=np.int32)
        pooled_text = audit.pool_sequences(text, lengths, "text")
        np.testing.assert_array_equal(pooled_text, np.array([[0., 1.], [0., 0.]], dtype=np.float32))

        audio = np.array([[[1., 3.], [3., 5.], [999., 999.]], [[7., 9.], [999., 999.], [999., 999.]]], dtype=np.float32)
        pooled_audio = audit.pool_sequences(audio, np.array([2, 1], dtype=np.int64), "audio")
        np.testing.assert_array_equal(pooled_audio, np.array([[2., 4.], [7., 9.]], dtype=np.float32))

    def test_fit_only_statistics_use_population_std_and_unit_tiny_scale(self):
        features = {
            "text": np.array([[0., 4.], [2., 4.], [10., 4.]], dtype=np.float32),
        }
        stats = audit.fit_only_statistics(features, [0, 1])
        np.testing.assert_array_equal(stats["text"]["mean"], [1., 4.])
        np.testing.assert_array_equal(stats["text"]["scale"], [1., 1.])
        self.assertEqual(stats["text"]["fit_indices"], [0, 1])

    def test_plateau_scheduler_replay_matches_patience_and_first_minimum(self):
        scores = [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
        rates = audit.replay_plateau_lr(scores, 1e-4, patience=3, factor=.5)
        self.assertEqual(rates[:4], [1e-4] * 4)
        self.assertEqual(rates[4], 1e-4)
        self.assertEqual(rates[8], 5e-5)
        self.assertEqual(audit.first_minimum_index([.4, .3, .3, .5]), 1)

    def test_array_difference_rejects_shape_mismatch(self):
        self.assertAlmostEqual(audit.max_abs_difference(np.array([1., 2.]), np.array([1., 2.000001])), 1e-6, places=12)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            audit.max_abs_difference(np.zeros(2), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
