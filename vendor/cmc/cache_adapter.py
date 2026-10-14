"""Canonical-cache adapter for the CMC pseudo-label five-class port."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from jevaffect.cache import load_cached_split


class CMCCacheError(ValueError):
    """Raised when the canonical cache cannot satisfy CMC input requirements."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _array_hash(value: np.ndarray) -> str:
    array = np.asarray(value)
    contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


class CMCFeatureSplit(Dataset):
    """One verified train or valid split with CMC-native sequence layout.

    `text_bert` is preferred and preserves the source CMC input `[N,3,T]`
    (input IDs, attention mask, token types). If token triplets are absent,
    precomputed contextual `text.npy` may be used through ContextualTextEncoder;
    this fallback cannot fine-tune BERT and is recorded as a distinct port mode.
    """

    def __init__(self, data: Mapping[str, Any], split: str) -> None:
        if split not in ("train", "valid"):
            raise CMCCacheError("CMC port reads only train and valid splits")
        self.split = split
        ids = data.get("ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids):
            raise CMCCacheError("cache ids must be a non-empty list[str]")
        if len(set(ids)) != len(ids):
            raise CMCCacheError("cache contains duplicate canonical IDs")
        self.ids = tuple(ids)

        labels = data.get("labels")
        label_valid = data.get("label_valid")
        raw_labels = data.get("raw_labels")
        if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype(np.int64):
            raise CMCCacheError("labels.npy must be int64 class targets")
        if not isinstance(label_valid, np.ndarray) or label_valid.dtype != np.dtype(np.bool_):
            raise CMCCacheError("label_valid.npy must be boolean")
        if labels.shape != (len(ids),) or label_valid.shape != (len(ids),):
            raise CMCCacheError("labels, label_valid, and IDs must align")
        if not bool(np.all(label_valid)) or np.any(labels < 0) or np.any(labels >= 5):
            raise CMCCacheError("all CMC train/valid targets must be known indices 0..4")
        if not isinstance(raw_labels, np.ndarray) or raw_labels.shape != (len(ids),):
            raise CMCCacheError("raw_labels.npy must align with IDs")
        if raw_labels.dtype.kind not in "fiu" or not np.isfinite(raw_labels).all():
            raise CMCCacheError("train/valid raw_labels must be finite numeric values")
        self.labels = labels
        self.raw_labels = raw_labels

        token_triplets = data.get("text_bert")
        contextual_text = data.get("text")
        if isinstance(token_triplets, np.ndarray):
            if token_triplets.ndim != 3 or token_triplets.shape[1] != 3:
                raise CMCCacheError("text_bert.npy must be [N,3,T] per MMSA/CMC contract")
            if token_triplets.shape[0] != len(ids) or token_triplets.dtype.kind not in "fiu":
                raise CMCCacheError("text_bert.npy must be numeric and align with IDs")
            if not np.isfinite(token_triplets).all():
                raise CMCCacheError("text_bert.npy contains non-finite token values")
            for channel, name in ((0, "input_ids"), (2, "token_type_ids")):
                values = np.asarray(token_triplets[:, channel, :])
                if not np.array_equal(values, np.rint(values)):
                    raise CMCCacheError("text_bert {} channel must contain integer-valued tokens".format(name))
            attention = np.asarray(token_triplets[:, 1, :])
            if np.any((attention != 0) & (attention != 1)):
                raise CMCCacheError("text_bert attention-mask channel must contain only 0/1")
            canonical_mask = data.get("text_mask")
            if isinstance(canonical_mask, np.ndarray) and not np.array_equal(
                attention.astype(np.bool_), canonical_mask.astype(np.bool_)
            ):
                raise CMCCacheError("text_bert attention mask disagrees with canonical text_mask")
            self.text_mode = "bert_token_triplets"
            self.text = token_triplets
            self.text_width = None
            self.text_sequence_length = int(token_triplets.shape[2])
        elif isinstance(contextual_text, np.ndarray):
            if contextual_text.ndim != 3 or contextual_text.shape[0] != len(ids):
                raise CMCCacheError("contextual text.npy must be [N,T,D] and align with IDs")
            if contextual_text.dtype.kind != "f" or not np.isfinite(contextual_text).all():
                raise CMCCacheError("contextual text.npy must be finite floating features")
            self.text_mode = "cached_contextual_text"
            self.text = contextual_text
            self.text_width = int(contextual_text.shape[2])
            self.text_sequence_length = int(contextual_text.shape[1])
        else:
            raise CMCCacheError(
                "CMC requires text_bert [N,3,T] or ready contextual text.npy [N,T,D]"
            )

        self.features: dict[str, np.ndarray] = {}
        for modality in ("audio", "vision"):
            feature = data.get(modality)
            if not isinstance(feature, np.ndarray) or feature.ndim != 3:
                raise CMCCacheError("{} must be a [N,T,D] feature array".format(modality))
            if feature.shape[0] != len(ids) or feature.dtype.kind != "f":
                raise CMCCacheError("{} must be floating and aligned with IDs".format(modality))
            if not np.isfinite(feature).all():
                raise CMCCacheError("{} contains non-finite values".format(modality))
            self.features[modality] = feature

        meta = data.get("meta")
        if not isinstance(meta, Mapping):
            raise CMCCacheError("cache reader must return verified meta.json as 'meta'")
        for key in ("source_sha256", "feature_dims", "sequence_lengths", "text_features_ready"):
            if key not in meta:
                raise CMCCacheError("cache meta.json is missing {}".format(key))
        self.meta = dict(meta)
        self.array_hashes = {
            "labels": _array_hash(labels),
            "raw_labels": _array_hash(raw_labels),
            "text": _array_hash(self.text),
            "audio": _array_hash(self.features["audio"]),
            "vision": _array_hash(self.features["vision"]),
        }

    def __len__(self) -> int:
        return len(self.ids)

    def close(self) -> None:
        """Release cache memmaps; useful for Windows fixture cleanup/reopen."""
        arrays = [self.text, self.labels, self.raw_labels, *self.features.values()]
        seen = set()
        for value in arrays:
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, np.memmap) and getattr(value, "_mmap", None) is not None:
                value._mmap.close()

    def __enter__(self) -> "CMCFeatureSplit":
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        del exc_type, exc_value, traceback_value
        self.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "text": torch.from_numpy(np.array(self.text[index], dtype=np.float32, copy=True)),
            "audio": torch.from_numpy(np.array(self.features["audio"][index], dtype=np.float32, copy=True)),
            "vision": torch.from_numpy(np.array(self.features["vision"][index], dtype=np.float32, copy=True)),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "raw_label": torch.tensor(float(self.raw_labels[index]), dtype=torch.float32),
            "id": self.ids[index],
        }


def load_cmc_train_valid(
    cache_dir: str | Path,
    *,
    data_loader=load_cached_split,
) -> tuple[CMCFeatureSplit, CMCFeatureSplit, dict[str, Any]]:
    """Use JevAffect's canonical reader; the test split is not requested."""
    root = Path(cache_dir).expanduser().resolve()
    train_data = data_loader(root, "train", require_text_features=False, require_raw_text=False)
    valid_data = data_loader(root, "valid", require_text_features=False, require_raw_text=False)
    train = CMCFeatureSplit(train_data, "train")
    valid = CMCFeatureSplit(valid_data, "valid")
    for key in ("source_sha256", "feature_dims", "sequence_lengths"):
        if train.meta.get(key) != valid.meta.get(key):
            raise CMCCacheError("train and valid meta.json {} differ".format(key))
    if train.text_mode != valid.text_mode:
        raise CMCCacheError("train and valid CMC text input representations differ")
    for modality in ("audio", "vision"):
        if train.features[modality].shape[1:] != valid.features[modality].shape[1:]:
            raise CMCCacheError("train/valid {} sequence shapes differ".format(modality))
    identity = {
        "source_sha256": train.meta["source_sha256"],
        "cache_meta_sha256": hashlib.sha256(_canonical_json(train.meta)).hexdigest(),
        "train_ids_sha256": hashlib.sha256(_canonical_json(list(train.ids))).hexdigest(),
        "valid_ids_sha256": hashlib.sha256(_canonical_json(list(valid.ids))).hexdigest(),
        "train_array_sha256": train.array_hashes,
        "valid_array_sha256": valid.array_hashes,
        "text_input_mode": train.text_mode,
    }
    return train, valid, identity
