import copy
import unittest

import numpy as np

from jevaffect.source_crossfit import make_folds
from scripts.prepare_anchor_calibration_tables import (
    _normalized_feature_hashes,
    make_calibration_split,
    merge_oof_records,
)


class AnchorTablePreparationTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"source{i:02d}/{clip:02d}" for i in range(15) for clip in range(2)]
        self.fold_manifest = make_folds(self.ids, fold_count=3, seed=20261012, inner_fraction=.1)
        self.targets = {sid: float(np.float32((i % 9 - 4) / 5)) for i, sid in enumerate(self.ids)}
        self.fold_records = {
            fold["fold"]: [
                {"id": sid, "target": self.targets[sid], "prediction": float(np.float32(i / 30))}
                for i, sid in enumerate(fold["oof_ids"])
            ]
            for fold in self.fold_manifest["folds"]
        }

    def test_merge_oof_orders_canonical_ids_and_checks_targets(self):
        merged = merge_oof_records(self.fold_records, self.fold_manifest["folds"], self.ids, self.targets)
        self.assertEqual([row["id"] for row in merged], self.ids)
        self.assertEqual(len(merged), len(self.ids))
        for row in merged:
            self.assertEqual(np.float32(row["target"]), np.float32(self.targets[row["id"]]))

    def test_merge_oof_rejects_duplicate_missing_and_wrong_target(self):
        duplicate = copy.deepcopy(self.fold_records)
        duplicate[0].append(copy.deepcopy(duplicate[0][0]))
        with self.assertRaises(ValueError):
            merge_oof_records(duplicate, self.fold_manifest["folds"], self.ids, self.targets)

        missing = copy.deepcopy(self.fold_records)
        missing[1].pop()
        with self.assertRaises(ValueError):
            merge_oof_records(missing, self.fold_manifest["folds"], self.ids, self.targets)

        wrong_target = copy.deepcopy(self.fold_records)
        wrong_target[2][0]["target"] = float(np.float32(self.targets[wrong_target[2][0]["id"]] + .2))
        with self.assertRaises(ValueError):
            merge_oof_records(wrong_target, self.fold_manifest["folds"], self.ids, self.targets)

    def test_fixed_calibration_split_is_source_disjoint_and_uses_canonical_indices(self):
        ids = [f"speaker{i:03d}/{clip:02d}" for i in range(129) for clip in range(2)]
        split = make_calibration_split(ids)
        repeated = make_calibration_split(ids)
        self.assertEqual(split, repeated)
        self.assertEqual(split["source_count"], 129)
        self.assertEqual(split["select_source_count"], 13)
        self.assertEqual(len(split["select_sources"]), 13)
        self.assertFalse(set(split["fit_sources"]) & set(split["select_sources"]))
        self.assertEqual(set(split["fit_ids"]) | set(split["select_ids"]), set(ids))
        self.assertEqual(set(split["fit_indices"]) | set(split["select_indices"]), set(range(len(ids))))
        self.assertFalse(set(split["fit_indices"]) & set(split["select_indices"]))
        self.assertEqual([ids[i] for i in split["fit_indices"]], split["fit_ids"])
        self.assertEqual([ids[i] for i in split["select_indices"]], split["select_ids"])

    def test_feature_hashes_normalize_only_the_cache_root(self):
        relative = {
            r"data\cache\meta.json": "a" * 64,
            r"data\cache\train\ids.json": "b" * 64,
            "train/text.npy": "c" * 64,
            "valid/audio_lengths.npy": "d" * 64,
        }
        absolute = {
            r"D:\project\data\cache\meta.json": "a" * 64,
            r"D:\project\data\cache\train\ids.json": "b" * 64,
            r"D:\project\data\cache\train\text.npy": "c" * 64,
            r"D:\project\data\cache\valid\audio_lengths.npy": "d" * 64,
        }
        self.assertEqual(_normalized_feature_hashes(relative), _normalized_feature_hashes(absolute))


if __name__ == "__main__":
    unittest.main()
