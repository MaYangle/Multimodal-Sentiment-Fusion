"""CPU-only boundary tests; all encoders and decoding are deterministic mocks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevaffect.raw_features import (  # noqa: E402
    ENCODER_PINS,
    EncoderBundle,
    RawFeatureError,
    RawVideoFeatureExtractor,
    conv_feature_geometry,
    conv_feature_timestamps,
    _load_dinov2_processor,
    verify_encoder_installation,
    whisper_chunk_ranges,
    write_raw_feature_cache,
)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        if kwargs.get("truncation") is False:
            return {"input_ids": [101, 7, 102]}
        return {
            "input_ids": np.asarray([[101, 7, 102]], dtype=np.int64),
            "attention_mask": np.ones((1, 3), dtype=np.int64),
        }


class FakeAudioProcessor:
    def __call__(self, waveform, **kwargs):
        return {"input_values": np.asarray(waveform, dtype=np.float32)[None, :]}


class FakeVisionProcessor:
    def __call__(self, images, **kwargs):
        return {"pixel_values": np.zeros((len(images), 3, 224, 224), dtype=np.float32)}


class FakeWhisperProcessor:
    def __init__(self):
        self.chunk_lengths = []
        self.attention_mask_requests = []

    def __call__(self, waveform, **kwargs):
        self.chunk_lengths.append(len(waveform))
        requested = kwargs.get("return_attention_mask", False)
        self.attention_mask_requests.append(requested)
        result = {"input_features": np.zeros((1, 80, 3000), dtype=np.float32)}
        if requested:
            result["attention_mask"] = np.ones((1, 3000), dtype=np.int64)
        return result

    def batch_decode(self, generated, **kwargs):
        return ["识别结果"]


class FakeModel:
    def __init__(self, hidden_size, *, kernels=None, strides=None, kind="text"):
        self.config = SimpleNamespace(hidden_size=hidden_size)
        if kernels is not None:
            self.config.conv_kernel = kernels
            self.config.conv_stride = strides
        self.kind = kind
        self.generate_tasks = []
        self.generate_kwargs = []
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.kind == "text":
            length = kwargs["input_ids"].shape[1]
            hidden = np.ones((1, length, self.config.hidden_size), dtype=np.float32)
        elif self.kind == "audio":
            sample_count = kwargs["input_values"].shape[-1]
            kernels = self.config.conv_kernel
            strides = self.config.conv_stride
            receptive, jump = conv_feature_geometry(kernels, strides)
            length = (sample_count - receptive) // jump + 1
            hidden = np.ones((1, length, self.config.hidden_size), dtype=np.float32)
        else:
            batch = kwargs["pixel_values"].shape[0]
            hidden = np.ones((batch, 197, self.config.hidden_size), dtype=np.float32)
        return SimpleNamespace(last_hidden_state=hidden)

    def generate(self, input_features, **kwargs):
        self.generate_tasks.append(kwargs.get("task"))
        self.generate_kwargs.append(kwargs)
        return np.asarray([[1]], dtype=np.int64)


class FakeCuda:
    def __init__(self):
        self.synchronize_calls = []
        self.reset_calls = []

    @staticmethod
    def is_available():
        return True

    def synchronize(self, device):
        self.synchronize_calls.append(device)

    @staticmethod
    def memory_allocated(device):
        return 1000

    @staticmethod
    def memory_reserved(device):
        return 2000

    def reset_peak_memory_stats(self, device):
        self.reset_calls.append(device)

    @staticmethod
    def max_memory_allocated(device):
        return 1600

    @staticmethod
    def max_memory_reserved(device):
        return 2500


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()

    @staticmethod
    def inference_mode():
        from contextlib import nullcontext
        return nullcontext()


def make_extractor(decoder, *, whisper=True, torch_module=None, device="cpu"):
    whisper_processor = FakeWhisperProcessor() if whisper else None
    whisper_model = FakeModel(1, kind="whisper") if whisper else None
    bundle = EncoderBundle(
        text_tokenizer=FakeTokenizer(),
        text_model=FakeModel(768, kind="text"),
        audio_processor=FakeAudioProcessor(),
        audio_model=FakeModel(768, kernels=[3, 3], strides=[2, 2], kind="audio"),
        vision_processor=FakeVisionProcessor(),
        vision_model=FakeModel(384, kind="vision"),
        whisper_processor=whisper_processor,
        whisper_model=whisper_model,
        torch_module=torch_module,
        device=device,
        provenance={"fixture": True, "fixture_kind": "cpu_mock_encoders_and_decoder"},
    )
    return RawVideoFeatureExtractor(bundle, decoder=decoder, device=device), bundle


class RawFeatureTests(unittest.TestCase):
    def test_convolution_timestamps_are_receptive_field_centers(self):
        self.assertEqual(conv_feature_geometry([3, 3], [2, 2]), (7, 4))
        np.testing.assert_allclose(
            conv_feature_timestamps(17, [3, 3], [2, 2], sample_rate=10),
            np.asarray([0.3, 0.7, 1.1]),
        )
        self.assertEqual(len(conv_feature_timestamps(6, [3, 3], [2, 2])), 0)

    def test_whisper_windows_cover_long_audio_without_dropping_tail(self):
        ranges = whisper_chunk_ranges(61 * 16_000 + 123)
        self.assertEqual(ranges, [(0, 480_000), (480_000, 960_000), (960_000, 976_123)])
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 61 * 16_000 + 123)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])))

    def test_dinov2_uses_explicit_slow_image_processor(self):
        class FakeImageProcessorLoader:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return object()

        _load_dinov2_processor(FakeImageProcessorLoader, "local/dinov2-small")
        args, kwargs = FakeImageProcessorLoader.calls[-1]
        self.assertEqual(args, ("local/dinov2-small",))
        self.assertIs(kwargs["use_fast"], False)
        self.assertIs(kwargs["local_files_only"], True)

    def test_verified_local_encoder_requires_complete_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "mbert"
            folder.mkdir()
            pin = dict(ENCODER_PINS["mbert"])
            weight = folder / pin["weight"]
            weight.write_bytes(b"mock-weight")
            import hashlib
            digest = hashlib.sha256(b"mock-weight").hexdigest()
            pin["sha256"] = digest
            for name in pin["files"]:
                path = folder / name
                if name == pin["weight"]:
                    continue
                path.write_bytes(b"mock-" + name.encode("ascii"))
            (folder / "provenance.json").write_text(json.dumps({
                "repo": pin["repo"], "revision": pin["revision"], "weight": pin["weight"],
                "sha256": digest, "files": pin["files"], "verified": True,
            }), encoding="utf-8")
            self.assertTrue(verify_encoder_installation("mbert", folder, pin)["verified"])
            weight.write_bytes(b"tampered-weight")
            with self.assertRaisesRegex(RawFeatureError, "weight SHA-256 mismatch"):
                verify_encoder_installation("mbert", folder, pin)

    def test_extract_keeps_sequences_masks_and_video_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"mock video source")
            waveform = np.linspace(-0.1, 0.1, 400, dtype=np.float32)
            frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
            times = np.asarray([0.0, 0.5], dtype=np.float64)

            def decoder(path, **kwargs):
                return {
                    "audio": waveform,
                    "frames": frames,
                    "frame_times": times,
                    "metadata": {
                        "source_sha256": "a" * 64,
                        "audio_status": "present",
                        "video_status": "present",
                        "frame_timestamps_kind": "ffmpeg_resampled_output_timeline",
                    },
                }

            extractor, bundle = make_extractor(decoder)
            result = extractor.extract_record({"id": "vid/clip", "video_path": str(video), "transcript": "你好"})
            arrays = result["arrays"]
            self.assertEqual(arrays["text"].shape, (3, 768))
            self.assertEqual(arrays["audio"].shape, (99, 768))
            self.assertEqual(arrays["vision"].shape, (2, 384))
            self.assertTrue(arrays["text_mask"].all())
            self.assertTrue(arrays["audio_mask"].all())
            self.assertTrue(arrays["vision_mask"].all())
            np.testing.assert_array_equal(arrays["vision_times"], times)
            self.assertEqual(bundle.whisper_model.calls, 0)
            self.assertTrue(result["metadata"]["fixture"])
            self.assertFalse(result["metadata"]["model_results"])
            self.assertEqual(result["metadata"]["prediction"], None)
            timing = result["metadata"]["performance"]
            self.assertGreaterEqual(timing["record_wall_seconds"], timing["decode_wall_seconds"])
            self.assertGreaterEqual(timing["feature_pipeline_wall_seconds"], 0.0)
            self.assertFalse(timing["feature_pipeline_gpu_synchronized"])
            self.assertFalse(timing["gpu_memory"]["measured"])

    def test_absent_modalities_are_empty_and_explicit(self):
        def decoder(path, **kwargs):
            return {
                "audio": np.empty(0, dtype=np.float32),
                "frames": np.empty((0, 224, 224, 3), dtype=np.uint8),
                "frame_times": np.empty(0, dtype=np.float64),
                "metadata": {"source_sha256": "b" * 64, "audio_status": "missing_stream",
                             "video_status": "missing_stream"},
            }

        extractor, _ = make_extractor(decoder)
        result = extractor.extract_record({"id": "vid/no-stream", "video_path": "unused.mp4"})
        arrays = result["arrays"]
        self.assertEqual(arrays["text"].shape, (0, 768))
        self.assertEqual(arrays["audio"].shape, (0, 768))
        self.assertEqual(arrays["vision"].shape, (0, 384))
        self.assertFalse(result["metadata"]["modalities"]["text"]["valid"])
        self.assertFalse(result["metadata"]["modalities"]["audio"]["valid"])
        self.assertFalse(result["metadata"]["modalities"]["vision"]["valid"])
        self.assertEqual(result["metadata"]["asr"]["status"], "no_audio_for_asr")

    def test_long_asr_calls_transcribe_for_every_chunk(self):
        extractor, bundle = make_extractor(lambda *args, **kwargs: {})
        result = extractor._transcribe(np.zeros(30 * 16_000 + 1, dtype=np.float32), "zh")
        self.assertEqual(bundle.whisper_processor.chunk_lengths, [480_000, 1])
        self.assertEqual(bundle.whisper_processor.attention_mask_requests, [True, True])
        self.assertEqual(bundle.whisper_model.generate_tasks, ["transcribe", "transcribe"])
        for kwargs in bundle.whisper_model.generate_kwargs:
            self.assertIn("attention_mask", kwargs)
            self.assertEqual(kwargs["attention_mask"].shape, (1, 3000))
            self.assertEqual(kwargs["task"], "transcribe")
            self.assertEqual(kwargs["language"], "zh")
            self.assertIsNone(kwargs["forced_decoder_ids"])
        self.assertEqual(result["task"], "transcribe")
        self.assertEqual(result["chunks"][-1]["end_seconds"], (30 * 16_000 + 1) / 16_000)

    def test_cuda_latency_and_peak_memory_use_synchronized_cpu_mocks(self):
        video = Path("mock-clip.mp4")
        audio = np.linspace(-0.1, 0.1, 400, dtype=np.float32)
        frames = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        fake_cuda_torch = FakeTorch()

        def decoder(path, **kwargs):
            return {
                "audio": audio,
                "frames": frames,
                "frame_times": np.asarray([0.25], dtype=np.float64),
                "metadata": {"source_sha256": "d" * 64},
            }

        extractor, _ = make_extractor(
            decoder, torch_module=fake_cuda_torch, device="cuda:0", whisper=False
        )
        result = extractor.extract_record({
            "id": "timed-mock", "video_path": str(video), "transcript": "mock"
        })
        performance = result["metadata"]["performance"]
        self.assertTrue(performance["feature_pipeline_gpu_synchronized"])
        self.assertGreaterEqual(len(fake_cuda_torch.cuda.synchronize_calls), 3)
        self.assertEqual(fake_cuda_torch.cuda.reset_calls, ["cuda:0"])
        self.assertTrue(performance["gpu_memory"]["measured"])
        self.assertEqual(performance["gpu_memory"]["peak_allocated_bytes"], 1600)
        self.assertEqual(performance["gpu_memory"]["peak_allocated_delta_bytes"], 600)
        self.assertEqual(performance["gpu_memory"]["peak_reserved_bytes"], 2500)
        self.assertTrue(result["metadata"]["fixture"])
        self.assertFalse(result["metadata"]["model_results"])

    def test_written_mock_bundle_is_marked_fixture_not_model_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"mock")

            def decoder(path, **kwargs):
                return {
                    "audio": np.zeros(20, dtype=np.float32),
                    "frames": np.zeros((0, 224, 224, 3), dtype=np.uint8),
                    "frame_times": np.empty(0, dtype=np.float64),
                    "metadata": {"source_sha256": "c" * 64},
                }

            extractor, _ = make_extractor(decoder, whisper=False)
            output = Path(tmp) / "features"
            manifest = Path(tmp) / "records.jsonl"
            manifest.write_text(json.dumps({"id": "sample-1", "video_path": str(video)}) + "\n",
                                 encoding="utf-8")
            contract = write_raw_feature_cache(
                [{"id": "sample-1", "video_path": str(video)}], output, extractor,
                input_manifest_path=manifest,
            )
            self.assertTrue(contract["fixture_results"])
            self.assertFalse(contract["model_results"])
            npz = next((output / "records").glob("*.npz"))
            with np.load(npz, allow_pickle=False) as values:
                self.assertEqual(values["text"].shape, (0, 768))
                self.assertEqual(values["audio"].shape[1], 768)
            row = json.loads((output / "index.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["metadata"]["fixture"])
            self.assertFalse(row["metadata"]["model_results"])


if __name__ == "__main__":
    unittest.main()
