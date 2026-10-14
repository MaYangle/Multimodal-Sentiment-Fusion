import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from jevaffect.native_raw import read_contract, pad_native_features,pooled_standardized_features
from jevaffect.native_training import digest


class NativeRawTests(unittest.TestCase):
    def test_standardized_media_uses_cls_and_masked_mean_after_truncation(self):
        arrays={"text":np.array([[1.,3.],[9.,9.]],np.float32),"text_length":np.array(2),
                "audio":np.array([[1.,4.],[2.,7.],[3.,10.],[999.,999.]],np.float32),"audio_length":np.array(3)}
        stats={"text":{"mean":[0.,1.],"scale":[.5,2.]},"audio":{"mean":[1.,1.],"scale":[.5,2.]}}
        values=pooled_standardized_features(arrays,stats,{"text":(1,2),"audio":(1,2)})
        np.testing.assert_array_equal(values["text"],[[2.,1.]])
        np.testing.assert_array_equal(values["audio"],[[2.,3.]])
        self.assertEqual(int(values["audio_length"]),1)

    def test_standardized_missing_modality_stays_zero(self):
        arrays={"audio":np.full((4,2),999.,np.float32),"audio_length":np.array(0)}
        values=pooled_standardized_features(arrays,{"audio":{"mean":[5.,8.],"scale":[1.,2.]}},{"audio":(1,2)})
        np.testing.assert_array_equal(values["audio"],np.zeros((1,2),np.float32))
        self.assertEqual(int(values["audio_length"]),0)

    def test_masks_truncation_and_no_interpolation(self):
        x = np.arange(24, dtype=np.float32).reshape(6, 4)
        extracted = {"arrays": {"audio": x, "audio_mask": np.ones(6, np.bool_)}}
        arrays, audit = pad_native_features(extracted, {"audio": (3, 4)})
        np.testing.assert_array_equal(arrays["audio"], x[:3])
        self.assertEqual(int(arrays["audio_length"]), 3)
        self.assertEqual(audit["audio"]["dropped_steps"], 3)
        extracted["arrays"]["audio_mask"] = np.array([1, 0, 1, 1, 0, 0], np.bool_)
        with self.assertRaises(ValueError):
            pad_native_features(extracted, {"audio": (3, 4)})
        extracted["arrays"]["audio_mask"][:] = False
        with self.assertRaises(ValueError):
            pad_native_features(extracted, {"audio": (3, 4)})

    def test_feature_padding_and_nonfinite_rejected(self):
        x = np.ones((2, 4), np.float32)
        arrays, _ = pad_native_features({"arrays": {"vision": x, "vision_mask": np.ones(2, np.bool_)}}, {"vision": (5, 4)})
        self.assertEqual(int(arrays["vision_length"]), 2)
        np.testing.assert_array_equal(arrays["vision"][2:], np.zeros((3, 4), np.float32))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            pad_native_features({"arrays": {"vision": x, "vision_mask": np.ones(2, np.bool_)}}, {"vision": (5, 4)})

    def test_published_feature_only_contract_is_not_raw_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.json"; path.write_text(json.dumps({"dataset": "mosei"}))
            spec = {"dataset": "mosei", "text_pooling": "valid_mean", "metadata_sha256": digest(path)}
            with self.assertRaises(ValueError):
                read_contract(folder, spec)
            spec["metadata_sha256"] = "changed"
            with self.assertRaises(ValueError):
                read_contract(folder, spec)
