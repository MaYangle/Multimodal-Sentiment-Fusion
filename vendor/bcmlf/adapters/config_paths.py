"""Resolve the BC-MLF config's repo-relative local data/model/checkpoint roots."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


_TOKEN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ROOT_NAMES = {
    "BCMLF_DATA_ROOT": Path("data") / "mmsa",
    "BCMLF_MODEL_ROOT": Path("models"),
    "BCMLF_CHECKPOINT_ROOT": Path("checkpoints"),
}


def resolve_bcmlf_paths(
    value: Any,
    *,
    repo_root: str | os.PathLike[str] | None = None,
    roots: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Any:
    """Recursively expand configured local roots in a JSON-like structure.

    Defaults are relative to the repository containing this package. Callers
    may pass explicit `roots` or environment variables for a different layout.
    Unknown placeholders raise instead of falling through to a remote model ID.
    """
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else (
        Path(__file__).resolve().parents[3]
    )
    resolved_roots: dict[str, str] = {}
    provided = dict(roots or {})
    unknown_roots = set(provided) - set(_ROOT_NAMES)
    if unknown_roots:
        raise ValueError(f"Unknown BC-MLF root override(s): {sorted(unknown_roots)}")
    for name, relative in _ROOT_NAMES.items():
        raw = provided.get(name, os.environ.get(name))
        path = Path(raw).expanduser() if raw is not None else root / relative
        if "://" in str(path):
            raise ValueError(f"BC-MLF roots must be local paths, got {str(path)!r}")
        if not path.is_absolute():
            path = root / path
        resolved_roots[name] = str(path.resolve())

    def resolve(item: Any) -> Any:
        if isinstance(item, str):
            def substitute(match: re.Match[str]) -> str:
                name = match.group(1)
                if name not in resolved_roots:
                    raise ValueError(f"Unresolved BC-MLF path variable: {name}")
                return resolved_roots[name]

            result = _TOKEN.sub(substitute, item)
            if "${" in result:
                raise ValueError(f"Unresolved path placeholder in {result!r}")
            return result
        if isinstance(item, list):
            return [resolve(element) for element in item]
        if isinstance(item, tuple):
            return tuple(resolve(element) for element in item)
        if isinstance(item, dict):
            return {key: resolve(element) for key, element in item.items()}
        return item

    return resolve(value)
