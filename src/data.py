"""Load COIL-20, per-image pre-processing, pose-spaced train/test split."""
import numpy as np

from .config import NORMALISE, STEP


def load_coil20(normalise: bool = NORMALISE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load processed COIL-20 from DATA_DIR.

    Returns X (1440, 1024) float64, y (1440,) labels 0-19, pose (1440,) 0-71.
    """
    raise NotImplementedError


def make_split(pose: np.ndarray, offset: int, step: int = STEP) -> tuple[np.ndarray, np.ndarray]:
    """Pose-spaced split: training poses are those with pose % step == offset.

    Returns (train_idx, test_idx) as integer index arrays.
    """
    raise NotImplementedError
