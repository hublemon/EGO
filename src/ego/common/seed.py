"""Random seed helpers for reproducible experiments."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch RNGs.

    ``deterministic`` additionally requests cuDNN determinism, which can slow
    down training but makes results bit-reproducible across runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
