import csv
import json
import unittest
import shutil
import numpy as np
from tests import test_anchor_crossfit as anchor_fixtures
from jevaffect.native_training import NativeSplit,author_rows
from jevaffect.native_training import train as train_native_model
from jevaffect.native_predict import load_model
from scripts.prepare_native_pooled_standardization import prepare_cache


class NativePooledStandardizationTests(unittest.TestCase):
    def test_native_scores_masked_pooling_and_fit_statistics_preserved(self):
        fixture=anchor_fixtures.AnchorCrossfitTests(methodName="test_source_cohorts_are_disjoint_and_every_train_id_is_held_once")
        fixture.setUp()
        try:
            with fixture.csv.open(encoding="utf-8",newline="") as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
            for row in rows:
                if row["mode"]=="valid":row["label"]=".2"
            with fixture.csv.open("w",encoding="utf-8",newline="") as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            out=fixture.root/"normalized"
            metadata=prepare_cache(fixture.cache,fixture.csv,out,allow_fixture=True)
            parsed=author_rows(fixture.csv,"simsv2")
            self.assertFalse(metadata["test_exported"])
            self.assertFalse((out/"test").exists())
            self.assertEqual(metadata["counts"],{"train":24,"valid":12})
            stats=json.loads((out/"normalization.json").read_text())
            train=NativeSplit(out,"train",parsed,("text","audio","vision"),pooled=True)
            valid=NativeSplit(out,"valid",parsed,("text","audio","vision"),pooled=True)
            for modality in train.features:
                np.testing.assert_allclose(train.features[modality].mean(0),0,atol=1e-6)
                np.testing.assert_allclose(train.features[modality].std(0),1,atol=1e-6)
                self.assertEqual(stats[modality]["fit_indices"],list(range(24)))
                self.assertTrue(np.all(train.lengths[modality]==1))
            np.testing.assert_array_equal(valid.targets,np.full(12,.2,dtype=np.float32))
            self.assertEqual(train.ids,fixture.ids["train"])
            cfg={**fixture.config,"cache_dir":str(out),"run_root":str(fixture.root/"native-run"),
                "model":"early_fusion","max_epochs":1}
            train.lengths.clear();valid.lengths.clear()
            train_native_model(cfg,13)
            descriptor=fixture.root/"descriptor-only";descriptor.mkdir()
            shutil.copyfile(out/"metadata.json",descriptor/"metadata.json")
            spec=load_model(fixture.root/"native-run"/"early_fusion"/"seed_13"/"best.pt",descriptor)
            self.assertEqual(spec["shapes"],{m:(1,d) for m,d in metadata["dimensions"].items()})
            with self.assertRaises(ValueError):prepare_cache(fixture.cache,fixture.csv,out,allow_fixture=True)
            train.lengths.clear();valid.lengths.clear()
        finally:fixture.tearDown()


if __name__=="__main__":unittest.main()
