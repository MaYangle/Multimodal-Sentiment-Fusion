"""Train BC-MLF five-class fusion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.bcmlf_stage2 import main
if __name__ == "__main__":
    raise SystemExit(main())
