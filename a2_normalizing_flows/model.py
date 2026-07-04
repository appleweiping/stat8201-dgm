"""RealNVP normalizing flow (Dinh et al. 2017, "Density estimation using Real NVP").

A normalizing flow models data with an invertible map f between the data space and a
latent space with a simple base density (standard Gaussian). Because f is invertible
with a tractable Jacobian, the exact log-likelihood is available via change of variables:

    log p(x) = log p_z(f(x)) + log |det df/dx|

RealNVP builds f from affine coupling layers. Each coupling layer splits the input by a
binary mask into (x_a, x_b), leaves x_a unchanged, and transforms x_b as

    y_b = x_b * exp(s(x_a)) + t(x_a)

whose Jacobian is triangular, so log|det| = sum(s(x_a)). Alternating the mask lets every
dimension be transformed. We operate on flattened, dequantized, logit-transformed MNIST.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Small MLP producing the scale (s) and translation (t) of a coupling layer."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim * 2),
        )
        # Initialize last layer to ~identity transform for stable training.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor):
        s, t = self.net(x).chunk(2, dim=1)
        s = torch.tanh(s)  # bound log-scale for stability
        return s, t


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 256):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = MLP(dim, hidden)

    def forward(self, x: torch.Tensor):
        """Forward map x -> z, returning (z, log|det|)."""
        x_a = x * self.mask
        s, t = self.net(x_a)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        z = x_a + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor):
        """Inverse map z -> x for sampling."""
        z_a = z * self.mask
        s, t = self.net(z_a)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        x = z_a + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x


def _spatial_checkerboard(h: int = 28, w: int = 28) -> torch.Tensor:
    """2D checkerboard mask flattened to (h*w,). Respects image structure, unlike a
    simple index-parity mask, which gives RealNVP much better image samples."""
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return ((yy + xx) % 2).reshape(-1)


class RealNVP(nn.Module):
    def __init__(self, dim: int = 784, n_coupling: int = 6, hidden: int = 256):
        super().__init__()
        self.dim = dim
        base = _spatial_checkerboard(28, 28) if dim == 784 else (torch.arange(dim) % 2)
        masks = []
        for i in range(n_coupling):
            masks.append(base if i % 2 == 0 else 1 - base)
        self.layers = nn.ModuleList(
            [AffineCoupling(dim, m.float(), hidden) for m in masks]
        )

    def forward(self, x: torch.Tensor):
        """x -> z with total log|det|."""
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z: torch.Tensor):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact log p(x) under standard-normal base density (in the flow space)."""
        z, log_det = self.forward(x)
        log_pz = (-0.5 * (z ** 2 + math.log(2 * math.pi))).sum(dim=1)
        return log_pz + log_det

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        z = torch.randn(n, self.dim, device=device)
        x = self.inverse(z)
        return x  # still in logit space; caller applies sigmoid


# --- logit dequantization (standard preprocessing for flows on images) ---
_ALPHA = 0.05


def logit_transform(x: torch.Tensor):
    """Map pixels in [0,1] to R via dequantize + logit, returning (y, log|det| of preproc).

    x -> alpha + (1-2 alpha) x, then y = logit(x'). The Jacobian log-det is added to the
    model log-likelihood so reported bits/dim are for the original pixel space.
    """
    x = x * 255.0
    x = (x + torch.rand_like(x)) / 256.0  # uniform dequantization
    x = _ALPHA + (1 - 2 * _ALPHA) * x
    y = torch.log(x) - torch.log1p(-x)
    ldj = (
        torch.log(torch.tensor(1 - 2 * _ALPHA))
        - torch.log(x)
        - torch.log1p(-x)
    ).sum(dim=1)
    return y, ldj


def inverse_logit(y: torch.Tensor) -> torch.Tensor:
    """Map flow-space samples back to [0,1] pixels for visualization."""
    x = torch.sigmoid(y)
    x = (x - _ALPHA) / (1 - 2 * _ALPHA)
    return x.clamp(0, 1)
