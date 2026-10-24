"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "coil-20-proc"
RESULTS_DIR = ROOT / "results"
IMG_SIZE = 32        # 32x32 -> n = 1024
STEP = 6             # every 6th pose for training -> 12 per object
OFFSETS = range(6)   # 6 disjoint splits
SEED = 0             # only used for optional random-split experiment E5
NORMALISE = True     # per-image brightness normalisation (main experiments)
