"""Load COIL-20, per-image pre-processing, pose-spaced train/test split."""
import re

import numpy as np
from PIL import Image

from .config import DATA_DIR, IMG_SIZE, NORMALISE, STEP

N_OBJECTS = 20
N_POSES = 72
_NAME = re.compile(r"obj(\d+)__(\d+)\.png")


def load_coil20(normalise: bool = NORMALISE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load processed COIL-20 from DATA_DIR.

    Each image is resized to IMG_SIZE x IMG_SIZE (area averaging), scaled to
    [0, 1] and, if normalise, centred on its own mean and divided by its own
    L2 norm. Only per-image operations are applied here.

    Returns X (1440, 1024) float64, y (1440,) labels 0-19, pose (1440,) 0-71,
    sorted by (object, pose).
    """
    records = []
    for path in DATA_DIR.glob("*.png"):
        match = _NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected file in {DATA_DIR}: {path.name}")
        records.append((int(match[1]), int(match[2]), path))
    if len(records) != N_OBJECTS * N_POSES:
        raise ValueError(f"Expected {N_OBJECTS * N_POSES} images in {DATA_DIR}, found {len(records)}")
    records.sort()

    X = np.empty((len(records), IMG_SIZE * IMG_SIZE))
    y = np.empty(len(records), dtype=int)
    pose = np.empty(len(records), dtype=int)
    for i, (obj, p, path) in enumerate(records):
        with Image.open(path) as im:
            im = im.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BOX)
        X[i] = np.asarray(im, dtype=np.float64).ravel() / 255.0
        y[i] = obj - 1
        pose[i] = p

    if not (np.bincount(y, minlength=N_OBJECTS) == N_POSES).all():
        raise ValueError("Each object must have exactly 72 poses")
    if not all(np.array_equal(np.sort(pose[y == k]), np.arange(N_POSES)) for k in range(N_OBJECTS)):
        raise ValueError("Each object must have poses 0-71 exactly once")

    if normalise:
        X -= X.mean(axis=1, keepdims=True)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, y, pose


def make_split(pose: np.ndarray, offset: int, step: int = STEP) -> tuple[np.ndarray, np.ndarray]:
    """Pose-spaced split: training poses are those with pose % step == offset.

    Returns (train_idx, test_idx) as integer index arrays.
    """
    if not 0 <= offset < step:
        raise ValueError(f"offset must be in [0, {step}), got {offset}")
    train = (pose % step) == offset
    return np.where(train)[0], np.where(~train)[0]


def make_random_split(y: np.ndarray, n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random split with n_per_class training images per class (E5 only).

    Returns (train_idx, test_idx) as sorted integer index arrays.
    """
    rng = np.random.default_rng(seed)
    train = np.concatenate([rng.choice(np.flatnonzero(y == k), n_per_class, replace=False)
                            for k in np.unique(y)])
    return np.sort(train), np.setdiff1d(np.arange(len(y)), train)
