"""Variational Autoencoder (Kingma & Welling 2014, "Auto-Encoding Variational Bayes").

This is the foundational model of STAT8201: an MLP encoder/decoder pair trained by
maximizing the Evidence Lower BOund (ELBO) with the reparameterization trick.

  ELBO(x) = E_{q(z|x)}[ log p(x|z) ] - KL( q(z|x) || p(z) )

The decoder outputs Bernoulli logits over the (dynamically binarized) pixels, and the
prior p(z) = N(0, I). The KL between two Gaussians has a closed form.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    logits: torch.Tensor  # decoder Bernoulli logits, (B, 784)
    mu: torch.Tensor      # q(z|x) mean, (B, D)
    logvar: torch.Tensor  # q(z|x) log-variance, (B, D)
    z: torch.Tensor       # sampled latent, (B, D)


class VAE(nn.Module):
    def __init__(self, x_dim: int = 784, hidden: int = 400, latent: int = 20):
        super().__init__()
        self.x_dim = x_dim
        self.latent = latent

        # Encoder q(z|x)
        self.enc_fc1 = nn.Linear(x_dim, hidden)
        self.enc_mu = nn.Linear(hidden, latent)
        self.enc_logvar = nn.Linear(hidden, latent)

        # Decoder p(x|z)
        self.dec_fc1 = nn.Linear(latent, hidden)
        self.dec_out = nn.Linear(hidden, x_dim)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I). Differentiable in (mu, logvar)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        return self.dec_out(h)  # logits

    def forward(self, x: torch.Tensor) -> VAEOutput:
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return VAEOutput(logits=logits, mu=mu, logvar=logvar, z=z)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Draw z ~ N(0, I) and return decoded Bernoulli means, shaped (n, 1, 28, 28)."""
        z = torch.randn(n, self.latent, device=device)
        logits = self.decode(z)
        probs = torch.sigmoid(logits)
        return probs.view(n, 1, 28, 28)


def elbo(out: VAEOutput, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (negative ELBO, reconstruction term, KL term), each averaged over the batch.

    Reconstruction is the Bernoulli negative log-likelihood summed over pixels.
    KL( N(mu, sigma^2) || N(0, I) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)).
    """
    x = x.view(x.size(0), -1)
    # summed over 784 pixels, then mean over batch
    recon = F.binary_cross_entropy_with_logits(out.logits, x, reduction="none").sum(dim=1)
    kl = -0.5 * torch.sum(1 + out.logvar - out.mu.pow(2) - out.logvar.exp(), dim=1)
    neg_elbo = (recon + kl).mean()
    return neg_elbo, recon.mean(), kl.mean()
