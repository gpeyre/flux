from pathlib import Path
import sys

# Allow running `python main.py` without requiring `pip install -e .`.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamics_sim.app import main


if __name__ == "__main__":
    main()
