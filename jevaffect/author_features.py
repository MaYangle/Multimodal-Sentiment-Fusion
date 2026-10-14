"""Explicit MultiBench aligned-feature adapter; independent of old CLS caches."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from jevaffect.native_training import digest

MODALITIES = ("text", "audio", "vision")
COHORT = "mosei_multibench_aligned50"


def temporal_ids(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected source/start/end IDs")
    ids = []
    for record in values:
        if any(not str(v) or "|" in str(v) for v in record):
            raise ValueError("Malformed temporal ID")
        ids.append("|".join(map(str, record)))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate publisher temporal ID")
    return ids


def aligned_suffix(part):
    """Match public loader slicing/casting and audio -inf fix; never drop rows."""
    source = {m: np.asarray(part[m]) for m in MODALITIES}
    n, steps = source["text"].shape[:2]
    if any(x.ndim != 3 or x.shape[:2] != (n, steps) for x in source.values()):
        raise ValueError("Aligned feature shapes disagree")
    if np.any(source["text"].sum(axis=(1, 2)) == 0):
        raise ValueError("Unexpected zero-sum text; explicit cohort decision required")
    nonzero = np.any(source["text"] != 0, axis=-1)
    if not nonzero.any(1).all():
        raise ValueError("No valid text token")
    starts = np.argmax(nonzero, axis=1); lengths = (steps - starts).astype(np.int64)
    prepared = {}; counters = {}
    for m, x in source.items():
        negative = np.isneginf(x)
        if np.isnan(x).any() or np.isposinf(x).any() or (m != "audio" and negative.any()):
            raise ValueError("Nonfinite values outside the public audio -inf rule")
        counters[m] = {"audio_negative_inf_replaced": int(negative.sum()) if m == "audio" else 0,
                       "all_zero_published_records": int((x == 0).all(axis=(1, 2)).sum())}
        output = np.zeros(x.shape, np.float32)
        for i, start in enumerate(starts):
            suffix = x[i, start:]
            if m == "audio":
                suffix = np.where(np.isneginf(suffix), 0., suffix)
            output[i, :lengths[i]] = suffix.astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("Float32 conversion overflow")
        prepared[m] = output
    return prepared, lengths, starts.astype(np.int64), counters


class AuthorFeatureSplit(Dataset):
    def __init__(self, cache, split, modalities=MODALITIES, pooled=True, limit=None):
        if split not in ("train", "valid"):
            raise ValueError("Training adapter never opens test")
        cache = Path(cache); metadata_path = cache / "metadata.json"
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.meta.get("cohort") != COHORT or self.meta.get("text_pooling") != "valid_mean":
            raise ValueError("Wrong author-feature cohort or pooling contract")
        self.hashes = {str(metadata_path): digest(metadata_path)}; self._maps = []
        directory = cache / split
        def verify(path):
            key = path.relative_to(cache).as_posix(); actual = digest(path)
            if self.meta["artifact_sha256"].get(key) != actual:
                raise ValueError("Changed prepared author artifact: " + key)
            self.hashes[key] = actual
        id_path, target_path = directory / "ids.json", directory / "targets.npy"
        verify(id_path); verify(target_path)
        self.ids = json.loads(id_path.read_text()); self.targets = np.load(target_path, allow_pickle=False)
        if len(self.ids) != self.meta["split_counts"][split] or len(set(self.ids)) != len(self.ids) or self.targets.shape != (len(self.ids),) or not np.isfinite(self.targets).all() or np.any(np.abs(self.targets) > 3):
            raise ValueError("Invalid prepared IDs or native scores")
        self.features, self.lengths = {}, {}
        for m in modalities:
            fp, lp = directory / (m + ".npy"), directory / (m + "_lengths.npy")
            verify(fp); verify(lp)
            x = np.load(fp, mmap_mode="r", allow_pickle=False); lengths = np.load(lp, mmap_mode="r", allow_pickle=False)
            self._maps.extend((x, lengths))
            if x.shape != (len(self.ids), *self.meta["feature_shapes"][m]) or x.dtype != np.float32 or lengths.shape != (len(self.ids),) or lengths.dtype.kind not in "iu" or np.any(lengths < 1) or np.any(lengths > x.shape[1]) or not np.isfinite(x).all():
                raise ValueError("Invalid author sequence or mask")
            if any(np.any(x[i, int(n):] != 0) for i, n in enumerate(lengths)):
                raise ValueError("Prepared padding must be zero")
            self.lengths[m] = lengths
            self.features[m] = np.stack([x[i, :int(n)].mean(0) for i, n in enumerate(lengths)]).astype(np.float32) if pooled else x
        if limit is not None:
            if limit < 1:
                raise ValueError("Positive smoke limit required")
            self.ids, self.targets = self.ids[:limit], self.targets[:limit]
        self.pooled = pooled

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        features = {m: torch.from_numpy(np.array(x[index], copy=True)) for m, x in self.features.items()}
        if not self.pooled:
            features["lengths"] = {m: torch.tensor(int(x[index])) for m, x in self.lengths.items()}
        return features, torch.tensor(self.targets[index]), self.ids[index]

    def close(self):
        for array in self._maps:
            array._mmap.close()
        self._maps = []
