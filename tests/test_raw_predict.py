"""CPU-only prediction tests using mock encoders, tensors, checkpoint, and model."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevaffect.raw_predict import (  # noqa: E402
    ALMT_PADDING_BEHAVIOR,
    MODALITIES,
    RawPredictionError,
    RawVideoPredictor,
    _canonical_json,
    load_classifier_model,
    load_run_spec,
    write_prediction_bundle,
)


CLASSES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")


def make_run_files(root: Path, *, model_name="almt", bad_cache_class_order=False):
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    source_sha = "a" * 64
    cache_meta = {
        "schema_version": 1,
        "source_sha256": source_sha,
        "text_features_ready": True,
        "feature_dims": {"text": 2, "audio": 3, "vision": 4},
        "sequence_lengths": {"text": 4, "audio": 3, "vision": 2},
        "class_order": list(reversed(CLASSES)) if bad_cache_class_order else list(CLASSES),
        "split_paths": {"train": "splits/train", "valid": "splits/valid", "test": "splits/test"},
    }
    cache_path = root / "cache-meta.json"
    cache_path.write_text(json.dumps(cache_meta, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    config = {
        "dataset": "simsv2-pilot",
        "language": "zh",
        "seeds": [13, 42, 2026],
        "model": {"name": model_name, "params": {"dim": 8} if model_name == "almt" else {"pooling": "cls"}},
    }
    config_sha = hashlib.sha256(_canonical_json(config)).hexdigest()
    fingerprint = {
        "meta_sha256": cache_sha,
        "source_hash": source_sha,
        "train": {"ids_sha256": "b" * 64, "labels_sha256": "c" * 64},
        "valid": {"ids_sha256": "d" * 64, "labels_sha256": "e" * 64},
    }
    environment = {"device": "cpu", "torch": "mock"}
    identity_payload = {
        "config_sha256": config_sha,
        "cache_fingerprint": fingerprint,
        "implementation_sha256": "f" * 64,
        "environment": environment,
        "seed": 13,
    }
    identity = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()
    manifest = {
        "format_version": 1,
        "identity_sha256": identity,
        "config_sha256": config_sha,
        "cache_fingerprint": fingerprint,
        "implementation_sha256": "f" * 64,
        "environment": environment,
        "seed": 13,
        "model": model_name,
        "dataset": config["dataset"],
        "language": config["language"],
        "class_order": list(CLASSES),
        "padding_behavior": ALMT_PADDING_BEHAVIOR if model_name == "almt" else "Text classifier uses configured CLS or length-masked mean pooling.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"mock checkpoint placeholder")
    return run_dir, cache_path, identity


class FakeTorch:
    float32 = np.float32
    long = np.int64
    cuda = None

    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint
        self.load_calls = []

    def load(self, path, **kwargs):
        self.load_calls.append(kwargs)
        return self.checkpoint

    @staticmethod
    def as_tensor(value, *, dtype, device):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeModel:
    def __init__(self, logits=None):
        self.logits = np.asarray(logits if logits is not None else [[0, 0, 0, 0, 0]], dtype=np.float32)
        self.inputs = None
        self.loaded_state = None
        self.strict = None
        self.evaluated = False
        self.device = None
        self.calls = 0

    def load_state_dict(self, state, *, strict):
        self.loaded_state = state
        self.strict = strict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, inputs):
        self.calls += 1
        self.inputs = inputs
        return self.logits


class FakeFeatureExtractor:
    def __init__(self, *, missing=()):
        self.missing = set(missing)

    def extract_record(self, record):
        arrays = {
            "text": np.arange(12, dtype=np.float32).reshape(6, 2),
            "text_mask": np.ones(6, dtype=np.bool_),
            "audio": np.arange(6, dtype=np.float32).reshape(2, 3),
            "audio_mask": np.ones(2, dtype=np.bool_),
            "audio_times": np.asarray([0.1, 0.2], dtype=np.float64),
            "vision": np.ones((4, 4), dtype=np.float32),
            "vision_mask": np.ones(4, dtype=np.bool_),
            "vision_times": np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float64),
        }
        modalities = {}
        for modality in MODALITIES:
            if modality in self.missing:
                arrays[modality] = np.empty((0, {"text": 2, "audio": 3, "vision": 4}[modality]), dtype=np.float32)
                arrays[modality + "_mask"] = np.empty(0, dtype=np.bool_)
            modalities[modality] = {"valid": modality not in self.missing, "status": "missing" if modality in self.missing else "present"}
        return {
            "arrays": arrays,
            "metadata": {
                "fixture": True,
                "model_results": False,
                "modalities": modalities,
                "preprocessing": {"video_frame_timestamps_kind": "mock output timeline"},
                "performance": {"record_wall_seconds": 0.01},
            },
        }


def make_prediction(sample_id, probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    index = int(np.argmax(values))
    max_probability = float(values[index])
    return {
        "ID": sample_id,
        "Label": CLASSES[index],
        "probabilities": dict(zip(CLASSES, (float(v) for v in values))),
        "argmax": CLASSES[index],
        "max_probability": max_probability,
        "choice_confidence": (max_probability - 0.2) / 0.8,
    }


def make_predictor(root: Path, *, model_name="almt", missing=(), logits=None):
    run_dir, cache_path, _ = make_run_files(root, model_name=model_name)
    spec = load_run_spec(run_dir, cache_path, class_order=CLASSES)
    model = FakeModel(logits)
    predictor = RawVideoPredictor(
        spec,
        model,
        FakeFeatureExtractor(missing=missing),
        class_order=CLASSES,
        low_confidence_threshold=0.1,
        torch_module=FakeTorch(),
        device="cpu",
        prediction_factory=make_prediction,
    )
    return predictor, model, spec


class RawPredictTests(unittest.TestCase):
    def test_run_identity_and_cache_shape_contract_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, cache_path, identity = make_run_files(Path(tmp))
            spec = load_run_spec(run_dir, cache_path, class_order=CLASSES)
            self.assertEqual(spec.identity_sha256, identity)
            self.assertEqual(spec.required_modalities, MODALITIES)
            self.assertEqual(spec.sequence_lengths, {"text": 4, "audio": 3, "vision": 2})

    def test_cache_class_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, cache_path, _ = make_run_files(Path(tmp), bad_cache_class_order=True)
            with self.assertRaisesRegex(RawPredictionError, "cache class_order"):
                load_run_spec(run_dir, cache_path, class_order=CLASSES)

    def test_checkpoint_load_uses_weights_only_and_matching_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, cache_path, identity = make_run_files(Path(tmp))
            spec = load_run_spec(run_dir, cache_path, class_order=CLASSES)
            checkpoint = {
                "format_version": 1,
                "identity_sha256": identity,
                "seed": 13,
                "model_state": {"mock-weight": np.ones(1, dtype=np.float32)},
            }
            torch = FakeTorch(checkpoint)
            model = FakeModel()
            factory_calls = []

            def factory(name, dims, lengths, params):
                factory_calls.append((name, dims, lengths, params))
                return model

            loaded_model, _, timing = load_classifier_model(
                spec, torch_module=torch, model_factory=factory, device="cpu"
            )
            self.assertIs(loaded_model, model)
            self.assertEqual(torch.load_calls, [{"map_location": "cpu", "weights_only": True}])
            self.assertEqual(len(factory_calls), 1)
            self.assertTrue(model.evaluated)
            self.assertIs(model.strict, True)
            self.assertGreaterEqual(timing["total_classifier_initialization_wall_seconds"], 0.0)

    def test_checkpoint_identity_mismatch_fails_before_model_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, cache_path, _ = make_run_files(Path(tmp))
            spec = load_run_spec(run_dir, cache_path, class_order=CLASSES)
            torch = FakeTorch({
                "format_version": 1,
                "identity_sha256": "0" * 64,
                "seed": 13,
                "model_state": {"weight": np.ones(1, dtype=np.float32)},
            })
            factory_calls = []
            with self.assertRaisesRegex(RawPredictionError, "identity differs"):
                load_classifier_model(
                    spec,
                    torch_module=torch,
                    model_factory=lambda *args: factory_calls.append(args),
                )
            self.assertEqual(factory_calls, [])
            self.assertTrue(torch.load_calls[0]["weights_only"])

    def test_weights_only_unsupported_fails_without_unsafe_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, cache_path, _ = make_run_files(Path(tmp))
            spec = load_run_spec(run_dir, cache_path, class_order=CLASSES)

            class UnsupportedSafeLoad(FakeTorch):
                def load(self, path, **kwargs):
                    self.load_calls.append(kwargs)
                    raise TypeError("weights_only unsupported")

            torch = UnsupportedSafeLoad()
            with self.assertRaisesRegex(RawPredictionError, "refusing unsafe fallback"):
                load_classifier_model(spec, torch_module=torch, model_factory=lambda *args: FakeModel())
            self.assertEqual(len(torch.load_calls), 1)
            self.assertIs(torch.load_calls[0]["weights_only"], True)

    def test_almt_pads_and_truncates_using_cache_caps_and_emits_five_probs(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictor, model, _ = make_predictor(Path(tmp), logits=[[0, 0, 0, 0, 0]])
            result = predictor.predict_record({"id": "clip-1", "video_path": "mock.mp4"})
            self.assertEqual(model.inputs["text"].shape, (1, 4, 2))
            self.assertEqual(model.inputs["audio"].shape, (1, 3, 3))
            self.assertEqual(model.inputs["vision"].shape, (1, 2, 4))
            np.testing.assert_array_equal(model.inputs["audio"][0, 2], np.zeros(3))
            self.assertEqual(result["modality_lengths"]["text"]["truncated_valid_positions"], 2)
            self.assertEqual(result["modality_lengths"]["audio"]["right_padding_positions"], 1)
            self.assertEqual(result["modality_lengths"]["vision"]["truncated_valid_positions"], 2)
            self.assertEqual(result["timestamp_references"]["audio"]["seconds"], [0.1, 0.2])
            self.assertEqual(result["timestamp_references"]["vision"]["seconds"], [0.0, 0.5])
            self.assertEqual(result["Label"], "SNEG")
            self.assertEqual(result["argmax"], "SNEG")
            self.assertEqual(list(result["probabilities"]), list(CLASSES))
            self.assertTrue(all(math.isfinite(p) for p in result["probabilities"].values()))
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
            self.assertTrue(result["low_confidence"])
            self.assertTrue(result["fixture"])
            self.assertFalse(result["model_results"])
            self.assertFalse(result["attention_weights_exported"])
            self.assertFalse(result["attention_interpretability_claim"])

    def test_almt_fails_clearly_when_required_modality_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictor, model, _ = make_predictor(Path(tmp), missing=("audio",))
            with self.assertRaisesRegex(RawPredictionError, "required audio modality is invalid"):
                predictor.predict_record({"id": "clip-no-audio", "video_path": "mock.mp4"})
            self.assertEqual(model.calls, 0)

    def test_raw_text_encoder_cap_must_match_training_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, model, spec = make_predictor(Path(tmp))

            class CappedFeatureExtractor(FakeFeatureExtractor):
                max_text_tokens = 256

            with self.assertRaisesRegex(RawPredictionError, "differs from trained cache cap"):
                RawVideoPredictor(
                    spec,
                    model,
                    CappedFeatureExtractor(),
                    class_order=CLASSES,
                    low_confidence_threshold=0.1,
                    torch_module=FakeTorch(),
                    device="cpu",
                    prediction_factory=make_prediction,
                )

    def test_text_model_only_requires_text_and_full_prediction_stays_separate_from_low_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, cache_path, _ = make_run_files(Path(tmp), model_name="text_mlp")
            spec = load_run_spec(run_dir, cache_path, class_order=CLASSES)
            model = FakeModel([[0, 1, 2, 3, 4]])
            predictor = RawVideoPredictor(
                spec,
                model,
                FakeFeatureExtractor(missing=("audio", "vision")),
                class_order=CLASSES,
                low_confidence_threshold=1.0,
                torch_module=FakeTorch(),
                device="cpu",
                prediction_factory=make_prediction,
            )
            result = predictor.predict_record({"id": "text-only", "video_path": "mock.mp4"})
            self.assertEqual(set(model.inputs), {"text", "lengths"})
            self.assertEqual(result["Label"], "SPOS")
            self.assertTrue(result["low_confidence"])
            self.assertFalse(result["modality_lengths"]["audio"]["used_for_model"])

    def test_writer_separates_low_confidence_and_marks_pilot_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictor, _, spec = make_predictor(root, logits=[[0, 0, 0, 0, 0]])
            input_manifest = root / "records.jsonl"
            input_manifest.write_text(json.dumps({"id": "clip-out", "video_path": "mock.mp4"}) + "\n", encoding="utf-8")
            output = root / "predictions"
            info = write_prediction_bundle(
                [{"id": "clip-out", "video_path": "mock.mp4"}],
                output,
                predictor,
                input_manifest_path=input_manifest,
                classifier_initialization={"total_classifier_initialization_wall_seconds": 0.1},
                encoder_initialization={"raw_encoder_load_wall_seconds": 0.2},
            )
            self.assertFalse(info["formal_benchmark"])
            self.assertTrue(info["fixture_results"])
            self.assertFalse(info["model_results"])
            prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
            low = json.loads((output / "low_confidence.jsonl").read_text(encoding="utf-8"))
            timing = json.loads((output / "timings.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(prediction["ID"], "clip-out")
            self.assertEqual(low["ID"], "clip-out")
            self.assertGreaterEqual(timing["whole_raw_to_prediction_wall_seconds"], 0.0)
            self.assertTrue((output / "run_info.json").is_file())


if __name__ == "__main__":
    unittest.main()
