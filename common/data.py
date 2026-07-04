"""MNIST data loading, shared across all assignments.

All models in this repo train on MNIST at CPU-modest scale. This module keeps the
download/transform logic in one place so every assignment sees identical data.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Repo-root/data — .gitignored, downloaded at runtime.
_DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _base_transform() -> transforms.Compose:
    # ToTensor already maps pixels to [0, 1].
    return transforms.Compose([transforms.ToTensor()])


def get_mnist(
    root: str = _DEFAULT_ROOT,
    train: bool = True,
    subset: int | None = None,
) -> datasets.MNIST:
    """Return the MNIST dataset (pixels in [0, 1]), downloading on first use.

    Args:
        root: where to cache the raw data.
        train: train vs. test split.
        subset: if given, keep only the first ``subset`` examples (for fast CPU runs).
    """
    ds = datasets.MNIST(root=root, train=train, download=True, transform=_base_transform())
    if subset is not None:
        ds = Subset(ds, list(range(min(subset, len(ds)))))
    return ds


def get_mnist_loaders(
    batch_size: int = 128,
    root: str = _DEFAULT_ROOT,
    train_subset: int | None = None,
    test_subset: int | None = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for MNIST."""
    train_ds = get_mnist(root=root, train=True, subset=train_subset)
    test_ds = get_mnist(root=root, train=False, subset=test_subset)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, test_loader


def dynamically_binarize(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample binary pixels ~ Bernoulli(x). Standard for binary-MNIST likelihood models."""
    return (torch.rand(x.shape, generator=generator, device=x.device) < x).float()
