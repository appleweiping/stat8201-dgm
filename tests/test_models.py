"""Correctness tests for the generative models: invertibility, log-dets, shapes, KL.

Run:  python -m pytest tests/ -q
These are the grad-check / consistency checks that stand in for the (nonexistent)
course autograder in this PhD seminar.
"""
import math

import torch

from a1_vae.model import VAE, elbo
from a2_normalizing_flows.model import RealNVP, logit_transform, _ALPHA


def test_vae_shapes_and_kl_nonneg():
    torch.manual_seed(0)
    m = VAE(latent=8)
    x = torch.rand(16, 1, 28, 28)
    out = m(x)
    assert out.logits.shape == (16, 784)
    neg_elbo, recon, kl = elbo(out, x)
    assert kl.item() >= 0.0  # KL is non-negative
    assert torch.isfinite(neg_elbo)


def test_vae_kl_zero_at_standard_normal():
    """KL(q||p)=0 when q is exactly N(0, I) (mu=0, logvar=0)."""
    torch.manual_seed(0)
    m = VAE(latent=8)
    x = torch.rand(4, 1, 28, 28)
    out = m(x)
    out.mu.zero_()
    out.logvar.zero_()
    _, _, kl = elbo(out, x)
    assert abs(kl.item()) < 1e-5


def test_realnvp_invertible():
    torch.manual_seed(0)
    m = RealNVP(dim=784, n_coupling=6, hidden=32)
    x = torch.randn(8, 784)
    z, _ = m.forward(x)
    x_rec = m.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-4)


def test_realnvp_logdet_matches_autograd():
    """Reported forward log|det J| equals the true Jacobian log-det (small dim)."""
    torch.manual_seed(0)
    m = RealNVP(dim=4, n_coupling=4, hidden=16)
    x = torch.randn(1, 4)
    J = torch.autograd.functional.jacobian(lambda t: m.forward(t)[0], x).reshape(4, 4)
    logdet_true = torch.logdet(J)
    _, logdet_model = m.forward(x)
    assert torch.allclose(logdet_true, logdet_model[0], atol=1e-4)


def test_logit_preprocess_logdet():
    """Preprocessing log-det formula matches autograd d/dv."""
    v = torch.tensor([0.3], requires_grad=True)
    xx = _ALPHA + (1 - 2 * _ALPHA) * v
    y = torch.log(xx) - torch.log1p(-xx)
    (g,) = torch.autograd.grad(y.sum(), v)
    ldj_true = torch.log(g)
    ldj_formula = math.log(1 - 2 * _ALPHA) - torch.log(xx) - torch.log1p(-xx)
    assert torch.allclose(ldj_true, ldj_formula[0], atol=1e-4)
