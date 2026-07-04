"""Correctness tests for PixelCNN (autoregressive masking) and DDPM (diffusion math)."""
import torch

from a4_autoregressive.model import PixelCNN
from a5_diffusion.model import DDPM, UNet, timestep_embedding


def test_pixelcnn_is_autoregressive():
    """Output at pixel (i,j) must not depend on input pixel (i,j) or any raster-future pixel."""
    torch.manual_seed(0)
    m = PixelCNN(hidden=16, n_res=2).eval()
    x = torch.rand(1, 1, 28, 28, requires_grad=True)
    out = m(x)
    g = torch.autograd.grad(out[0, 0, 5, 5], x)[0][0, 0]
    future = torch.zeros(28, 28, dtype=torch.bool)
    future[5, 5:] = True
    future[6:, :] = True
    assert g[future].abs().max().item() == 0.0  # no leakage
    assert g[~future].abs().max().item() > 0.0  # depends on the past


def test_pixelcnn_output_shape():
    m = PixelCNN(hidden=16, n_res=2)
    x = torch.rand(4, 1, 28, 28)
    assert m(x).shape == (4, 1, 28, 28)


def test_ddpm_alpha_bar_schedule():
    d = DDPM(UNet(base=8), T=100)
    assert bool((d.alpha_bar[1:] <= d.alpha_bar[:-1]).all())  # monotone decreasing
    assert d.alpha_bar[0].item() > 0.99  # barely noised at t=0
    assert d.alpha_bar[-1].item() < d.alpha_bar[0].item()


def test_ddpm_q_sample_closed_form():
    d = DDPM(UNet(base=8), T=100)
    x0 = torch.ones(2, 1, 28, 28)
    t = torch.zeros(2, dtype=torch.long)
    xt = d.q_sample(x0, t, torch.zeros_like(x0))
    assert torch.allclose(xt, d.sqrt_alpha_bar[0] * x0, atol=1e-6)


def test_timestep_embedding_shape():
    assert timestep_embedding(torch.arange(4), 128).shape == (4, 128)
    assert timestep_embedding(torch.arange(4), 127).shape == (4, 127)  # odd dim padded
