"""DCGAN generator/discriminator for 28x28 MNIST (Radford et al. 2016).

The seminar devotes several weeks to GANs (GANsition, Wasserstein GAN, improving GANs,
GAN theory). This module implements a small convolutional GAN and supports two training
objectives selected in train.py:

  * "bce"  -- the original minimax GAN (Goodfellow et al. 2014) with the non-saturating
             generator loss.
  * "wgan_gp" -- Wasserstein GAN with gradient penalty (Arjovsky et al. 2017;
             Gulrajani et al. 2017), the "Wasserstein GAN" week of the course.

Both share the same architecture; WGAN-GP simply drops the sigmoid on the critic and uses
a different loss + gradient penalty.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim: int = 64, ngf: int = 32):
        super().__init__()
        # z -> 7x7 -> 14x14 -> 28x28
        self.fc = nn.Linear(z_dim, ngf * 4 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),  # pixels in [0, 1]
        )
        self.z_dim = z_dim
        self.ngf = ngf

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), self.ngf * 4, 7, 7)
        return self.net(h)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        z = torch.randn(n, self.z_dim, device=device)
        return self.forward(z)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 32, use_sigmoid: bool = True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, ndf, 4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),  # 7x7
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(ndf * 2 * 7 * 7, 1)
        self.use_sigmoid = use_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        out = self.head(h)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out.squeeze(1)


def weights_init(m: nn.Module) -> None:
    """DCGAN weight init: conv/deconv ~ N(0, 0.02), batchnorm gamma ~ N(1, 0.02)."""
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cls and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)
