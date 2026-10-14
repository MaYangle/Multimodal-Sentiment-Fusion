"""Runtime tests for the adapter; skipped when PyTorch is not installed."""

from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None


PORT_ROOT = Path(__file__).resolve().parents[1]
if str(PORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PORT_ROOT))


class ConfigPathResolverTests(unittest.TestCase):
    def test_repo_relative_roots_are_expanded_recursively(self):
        from vendor.bcmlf.adapters.config_paths import resolve_bcmlf_paths

        with tempfile.TemporaryDirectory() as root:
            result = resolve_bcmlf_paths(
                {
                    "data": "${BCMLF_DATA_ROOT}/simsv2/features.pkl",
                    "lm": "${BCMLF_MODEL_ROOT}/gpt2-cn",
                },
                repo_root=root,
                roots={
                    "BCMLF_DATA_ROOT": "data/mmsa",
                    "BCMLF_MODEL_ROOT": "models",
                    "BCMLF_CHECKPOINT_ROOT": "checkpoints",
                },
            )
        self.assertTrue(Path(result["data"]).as_posix().endswith("data/mmsa/simsv2/features.pkl"))
        self.assertTrue(Path(result["lm"]).as_posix().endswith("models/gpt2-cn"))

    def test_unknown_path_variable_is_rejected(self):
        from vendor.bcmlf.adapters.config_paths import resolve_bcmlf_paths

        with self.assertRaises(ValueError):
            resolve_bcmlf_paths("${UNDECLARED_ROOT}/weights")


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this runtime")
class FiveClassAdapterTests(unittest.TestCase):
    def setUp(self):
        from vendor.bcmlf.adapters.five_class import (
            bcmlf_five_class_loss,
            build_causal_lm_targets,
            five_class_forward_view,
        )

        self.loss_fn = bcmlf_five_class_loss
        self.lm_targets_fn = build_causal_lm_targets
        self.view_fn = five_class_forward_view

    def _outputs(self):
        return {
            "task_logits": torch.randn(3, 5, requires_grad=True),
            "bn_logits": torch.randn(3, 2, 5, requires_grad=True),
            "text_logits": torch.randn(3, 5, requires_grad=True),
            "av_logits": torch.randn(3, 5, requires_grad=True),
            "Feature_bn": torch.randn(3, 4, requires_grad=True),
            "lm_logits": torch.randn(3, 6, 11, requires_grad=True),
            "Contribution_weights": torch.softmax(torch.randn(3, 3), dim=-1),
        }

    def test_loss_covers_all_heads_and_raw_label_contrast(self):
        outputs = self._outputs()
        targets = torch.tensor([0, 2, 4], dtype=torch.long)
        raw_y = torch.tensor([-1.0, 0.0, 1.0])
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])

        losses = self.loss_fn(
            outputs,
            targets,
            raw_y,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(
            set(losses),
            {
                "loss", "task_loss", "bn_loss", "av_loss", "text_loss",
                "lm_loss", "contrast_loss",
            },
        )
        losses["loss"].backward()
        for name in (
            "task_logits", "bn_logits", "text_logits", "av_logits",
            "Feature_bn", "lm_logits",
        ):
            self.assertIsNotNone(outputs[name].grad, name)

    def test_default_clm_requires_explicit_targets_or_input_ids(self):
        outputs = self._outputs()
        with self.assertRaisesRegex(ValueError, "use_clm=True requires"):
            self.loss_fn(
                outputs,
                torch.tensor([0, 2, 4], dtype=torch.long),
                torch.tensor([-1.0, 0.0, 1.0]),
            )

    def test_lm_shift_and_padding_match_source_trainer(self):
        input_ids = torch.tensor([[5, 6, 2, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        targets = self.lm_targets_fn(input_ids, attention_mask)
        torch.testing.assert_close(targets, torch.tensor([[6, 2, 0, -1]]))

    def test_rejects_non_five_class_branch_shape(self):
        outputs = self._outputs()
        outputs["av_logits"] = torch.randn(3, 4)
        with self.assertRaises(ValueError):
            self.loss_fn(
                outputs,
                torch.tensor([0, 2, 4], dtype=torch.long),
                torch.tensor([-1.0, 0.0, 1.0]),
                use_clm=False,
            )

    def test_forward_view_keeps_class_probability_and_branch_weight_separate(self):
        outputs = self._outputs()
        view = self.view_fn(outputs)
        self.assertEqual(tuple(view["class_probabilities_uncalibrated"].shape), (3, 5))
        torch.testing.assert_close(
            view["class_probabilities_uncalibrated"].sum(dim=-1),
            torch.ones(3),
        )
        self.assertEqual(tuple(view["branch_mixing_weights"].shape), (3, 3))
        self.assertEqual(tuple(view["branch_logits"]["fusion"].shape), (3, 5))


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this runtime")
class StrictAVCheckpointLoadTests(unittest.TestCase):
    def test_authored_bienc_model_imports_from_vendor_closure(self):
        from vendor.bcmlf.models.singleTask.BI_ENC import BI_ENC

        self.assertTrue(issubclass(BI_ENC, torch.nn.Module))

    def setUp(self):
        from vendor.bcmlf.models.singleTask.MMGPT import AV_Enc

        self.encoder_class = AV_Enc

    @staticmethod
    def _tiny_init(module, config):
        torch.nn.Module.__init__(module)
        module.proj = torch.nn.Linear(2, 3)

    def _save_checkpoint(self, path, *, omit=None, bad_shape=False):
        state = {
            "Model.proj.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
            "Model.proj.bias": torch.arange(3, dtype=torch.float32),
            "Model.clf.weight": torch.ones(1),
        }
        if omit is not None:
            state.pop(omit)
        if bad_shape:
            state["Model.proj.weight"] = torch.ones(4, 2)
        torch.save(state, path)

    def test_loads_complete_prefixed_state_and_ignores_only_extra_clf(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "bienc.pth"
            self._save_checkpoint(checkpoint)
            with patch.object(self.encoder_class, "__init__", self._tiny_init):
                model = self.encoder_class.from_pretrained(checkpoint, {})
        torch.testing.assert_close(
            model.proj.weight, torch.arange(6, dtype=torch.float32).reshape(3, 2)
        )

    def test_missing_target_key_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "missing.pth"
            self._save_checkpoint(checkpoint, omit="Model.proj.bias")
            with patch.object(self.encoder_class, "__init__", self._tiny_init):
                with self.assertRaisesRegex(RuntimeError, "missing Model-prefixed target keys"):
                    self.encoder_class.from_pretrained(checkpoint, {})

    def test_mismatched_shape_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "mismatch.pth"
            self._save_checkpoint(checkpoint, bad_shape=True)
            with patch.object(self.encoder_class, "__init__", self._tiny_init):
                with self.assertRaisesRegex(RuntimeError, "shape/type mismatches"):
                    self.encoder_class.from_pretrained(checkpoint, {})


if __name__ == "__main__":
    unittest.main()
