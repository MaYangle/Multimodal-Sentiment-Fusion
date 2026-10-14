import csv
import tempfile
import unittest
from pathlib import Path
from scripts.diagnose_native_annotations import load_train_annotations,summarize


class NativeAnnotationDiagnosticTests(unittest.TestCase):
    def test_nontrain_scores_are_never_parsed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"meta.csv"
            path.write_text("video_id,clip_id,mode,label,label_T,label_A,label_V\nv,1,train,0.2,0,0.2,-0.2\nv,2,valid,INVALID,INVALID,INVALID,INVALID\nv,3,test,INVALID,INVALID,INVALID,INVALID\n",encoding="utf-8")
            ann=load_train_annotations(path)
            self.assertEqual(set(ann),{"v/1"})
            ids,arrays,slices,summary=summarize(ann)
            self.assertEqual(summary["slice_counts"]["unimodal_sign_conflict"],1)
            self.assertEqual(summary["slice_counts"]["text_M_sign_conflict"],1)

    def test_missing_channels_do_not_become_negative_agreement(self):
        ann={"a/1":{"M":0.,"T":None,"A":0.,"V":0.},"a/2":{"M":0.,"T":0.,"A":0.,"V":0.}}
        _,_,_,s=summarize(ann)
        self.assertEqual(s["channels"]["T"]["missing"],1)
        self.assertEqual(s["slice_counts"]["unimodal_sign_agreement"],1)
        self.assertEqual(s["channel_M_agreement"]["T"]["paired_count"],1)

    def test_native_score_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"meta.csv"
            header="video_id,clip_id,mode,label,label_T,label_A,label_V\n"
            path.write_text(header+"v,1,train,0,2,0,0\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Invalid native"):
                load_train_annotations(path)
            row="v,1,train,0,0,0,0\n"
            path.write_text(header+row+row,encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                load_train_annotations(path)


if __name__ == "__main__":unittest.main()
