"""Train one cached-feature baseline config from the JevAffect project root."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set before importing the training module (and torch).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jevaffect.training import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
