"""Small shared helpers: seeding, thread control, image-grid saving."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_cpu(threads: int | None = None) -> None:
    """Constrain thread count for reproducible, polite CPU runs."""
    if threads is None:
        threads = int(os.environ.get("OMP_NUM_THREADS", "3"))
    torch.set_num_threads(threads)


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """Save a grid of images (N, C, H, W) in [0, 1] to ``path`` as PNG.

    Uses torchvision.utils.make_grid + a manual matplotlib save so we do not depend
    on PIL being importable in odd environments.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    images = images.detach().cpu().clamp(0, 1)
    grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
    np_grid = grid.permute(1, 2, 0).numpy()
    if np_grid.shape[2] == 1:
        np_grid = np_grid[:, :, 0]
        cmap = "gray"
    else:
        cmap = None
    h, w = np_grid.shape[0], np_grid.shape[1]
    fig = plt.figure(figsize=(w / 40.0, h / 40.0), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(np_grid, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.axis("off")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
