"""Denoising Diffusion Probabilistic Model (Ho et al. 2020) for MNIST.

A diffusion model defines a fixed forward process that gradually adds Gaussian noise to
an image over T steps,

    q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I),

and learns to reverse it. Ho et al. show the reverse process can be trained with the
simple objective of predicting the noise eps added at a random step t:

    L = E_{x_0, t, eps} || eps - eps_theta(x_t, t) ||^2 .

Sampling runs the learned reverse chain from x_T ~ N(0, I) back to x_0. The eps-network
here is a compact U-Net with sinusoidal timestep embeddings, sized for CPU training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps (Transformer-style)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.temb = nn.Linear(t_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.temb(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Small U-Net eps-predictor for 28x28 single-channel images."""

    def __init__(self, base: int = 32, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))

        self.down1 = ConvBlock(1, base, t_dim)          # 28
        self.down2 = ConvBlock(base, base * 2, t_dim)    # 14
        self.mid = ConvBlock(base * 2, base * 2, t_dim)  # 7
        self.up2 = ConvBlock(base * 4, base, t_dim)      # 14
        self.up1 = ConvBlock(base * 2, base, t_dim)      # 28
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        d1 = self.down1(x, t_emb)              # (B, base, 28, 28)
        d2 = self.down2(self.pool(d1), t_emb)  # (B, 2base, 14, 14)
        m = self.mid(self.pool(d2), t_emb)     # (B, 2base, 7, 7)
        u2 = F.interpolate(m, size=d2.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)  # (B, base, 14, 14)
        u1 = F.interpolate(u2, size=d1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)  # (B, base, 28, 28)
        return self.out(u1)


class DDPM(nn.Module):
    def __init__(self, model: UNet, T: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) using the closed form."""
        sab = self.sqrt_alpha_bar[t][:, None, None, None]
        somab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sab * x0 + somab * noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Simple DDPM training objective: MSE between true and predicted noise."""
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Ancestral sampling of the reverse process; returns images in [0, 1]."""
        self.model.eval()
        x = torch.randn(n, 1, 28, 28, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.model(x, t)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bar[i]
            coef = beta / self.sqrt_one_minus_alpha_bar[i]
            mean = (x - coef * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        # data was trained in [-1, 1]; map back to [0, 1]
        return ((x + 1) / 2).clamp(0, 1)
