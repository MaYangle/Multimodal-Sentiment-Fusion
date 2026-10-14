import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.prepare_native_timed_grid import interpolate, paired_grid, prepare_cache
from tests.test_native_sequence_standardization import _sequence_fixture


class NativeTimedGridTests(unittest.TestCase):
    def test_linear_seconds_and_overlap_no_index_stretch(self):
        a = np.array([[0.],[2.],[8.]],np.float32)
        at = np.array([0.,1.,4.])
        v = np.array([[4.],[16.]],np.float32)
        vt = np.array([1.,4.])
        ag,vg,q,valid = paired_grid(a,at,v,vt)
        self.assertTrue(valid)
        np.testing.assert_allclose(q,[1.6,2.2,2.8,3.4],atol=1e-12)
        np.testing.assert_allclose(ag[:,0],2*q)
        np.testing.assert_allclose(vg[:,0],4*q)
        np.testing.assert_allclose(interpolate(a,at,[0,4])[:,0],[0,8])
        with self.assertRaises(ValueError):
            interpolate(a,at,[-.1])

    def test_invalid_prefix_and_missing_overlap(self):
        a = np.ones((2,3),np.float32)
        v = np.ones((2,5),np.float32)
        for at,vt in (([0,1],[2,3]),([0],[0,1])):
            ag,vg,q,valid = paired_grid(a[:len(at)],np.array(at),v,np.array(vt))
            self.assertFalse(valid)
            self.assertEqual(ag.sum()+vg.sum()+q.sum(),0)
        with self.assertRaises(ValueError):
            paired_grid(a,np.array([1,0]),v,np.array([0,1]))

    def test_cache_native_labels_masks_padding_hash_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw,pooled,csv_path = _sequence_fixture(root)
            output = root/"timed"
            meta = prepare_cache(raw,csv_path,output,allow_fixture=True)
            self.assertEqual(meta["counts"],{"train":5,"valid":3})
            self.assertEqual(len(meta["artifact_sha256"]),20)
            self.assertEqual(meta["time_contract"]["corpus_physical_synchrony"],"UNVERIFIED")
            self.assertFalse((output/"test").exists())
            for split in ("train","valid"):
                presence = np.load(output/split/"presence.npy")
                valid = np.load(output/split/"pair_valid.npy")
                self.assertTrue(np.all(~valid | (presence[:,1]&presence[:,2])))
                for m in ("audio","vision"):
                    grid = np.load(output/split/(m+"_grid.npy"))
                    self.assertTrue(np.all(grid[~valid]==0))
                labels = json.loads((output/split/"raw_labels.json").read_text())
                self.assertEqual(len(labels),meta["counts"][split])
            # The producer has no TEST score parser; fixture includes an invalid
            # TEST string. Corrupt time bytes must fail before another export.
            path = raw/"splits/train/audio_times.npy"
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"SHA"):
                prepare_cache(raw,csv_path,root/"bad",allow_fixture=True)
            self.assertFalse((root/"bad").exists())
            with self.assertRaisesRegex(ValueError,"overwrite"):
                prepare_cache(raw,csv_path,output,allow_fixture=True)


if __name__ == "__main__":
    unittest.main()
