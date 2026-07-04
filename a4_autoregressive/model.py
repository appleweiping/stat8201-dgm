"""PixelCNN autoregressive model (van den Oord et al. 2016).

Autoregressive models factorize the image likelihood pixel-by-pixel in raster order:

    p(x) = prod_i p(x_i | x_{<i})

PixelCNN enforces this with *masked* convolutions: the first ("type A") mask blocks the
current pixel and all future pixels; deeper ("type B") masks allow the current pixel.
This gives a proper autoregressive receptive field while remaining fully convolutional,
so the full-image likelihood is computed in one forward pass (sampling is sequential).

We model binary MNIST (dynamically binarized), so each pixel is a Bernoulli whose logit
is the network output — an exact, tractable log-likelihood.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    """Conv2d with an autoregressive mask over the kernel.

    mask_type 'A' excludes the center pixel (used for the first layer so a pixel never
    sees itself); 'B' includes the center (used for subsequent layers).
    """

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        yc, xc = kh // 2, kw // 2
        # zero out "future" pixels in raster order
        self.mask[:, :, yc, xc + (mask_type == "B"):] = 0
        self.mask[:, :, yc + 1:, :] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class ResidualBlock(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", h, h // 2, 1),
            nn.ReLU(),
            MaskedConv2d("B", h // 2, h // 2, 3, padding=1),
            nn.ReLU(),
            MaskedConv2d("B", h // 2, h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PixelCNN(nn.Module):
    def __init__(self, hidden: int = 64, n_res: int = 5):
        super().__init__()
        self.hidden = hidden
        self.input_conv = MaskedConv2d("A", 1, hidden, 7, padding=3)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_res)])
        self.output = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", hidden, hidden, 1),
            nn.ReLU(),
            MaskedConv2d("B", hidden, 1, 1),  # per-pixel Bernoulli logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel Bernoulli logits, same spatial shape as x."""
        h = self.input_conv(x)
        for block in self.res_blocks:
            h = block(h)
        return self.output(h)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Ancestral sampling: fill pixels one at a time in raster order."""
        self.eval()
        x = torch.zeros(n, 1, 28, 28, device=device)
        for i in range(28):
            for j in range(28):
                logits = self.forward(x)
                probs = torch.sigmoid(logits[:, :, i, j])
                x[:, :, i, j] = torch.bernoulli(probs)
        return x
