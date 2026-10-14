import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import torch
from tests.test_native_sequence_standardization import _sequence_fixture, SHAPES
from scripts.prepare_native_sequence_standardization import prepare_cache, _close_dataset
from jevaffect.native_training import train,NativeSplit,author_rows,evaluate
from jevaffect.native_predict import load_model


class NativeSequenceInferenceTests(unittest.TestCase):
    def test_sequence_normalization_descriptor_retains_full_lengths_for_almt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw,pooled,csv=_sequence_fixture(root)
            normalized=root/'normalized'
            prepare_cache(raw,pooled,normalized,csv,allow_fixture=True)
            cfg=dict(dataset='simsv2',author_csv=str(csv),cache_dir=str(normalized),run_root=str(root/'runs'),
                     model='almt',seeds=[13,42,2026],device='cpu',batch_size=2,learning_rate=.0001,
                     weight_decay=.0001,max_epochs=1,patience=12)
            with contextlib.redirect_stdout(io.StringIO()):train(cfg,13)
            checkpoint=root/'runs/almt/seed_13/best.pt'
            spec=load_model(checkpoint,normalized,'cpu')
            self.assertEqual(spec['shapes'],SHAPES)
            data=NativeSplit(normalized,'valid',author_rows(csv,'simsv2'),('text','audio','vision'),pooled=False)
            _,actual=evaluate(spec['model'],torch.utils.data.DataLoader(data,batch_size=2),'cpu','simsv2')
            stored=json.loads((checkpoint.parent/'valid_predictions.json').read_text())
            self.assertEqual(actual,stored)
            _close_dataset(data)


if __name__=='__main__':unittest.main()
