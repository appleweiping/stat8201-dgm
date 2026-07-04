"""Train the VAE on dynamically binarized MNIST and save generated samples.

Run:
    python -m a1_vae.train --epochs 10 --results results/a1_vae

Produces:
    results/a1_vae/samples.png        -- images decoded from z ~ N(0, I)
    results/a1_vae/reconstructions.png-- top row inputs, bottom row reconstructions
    results/a1_vae/metrics.json       -- test neg-ELBO and IWAE log-likelihood bound
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from common.data import dynamically_binarize, get_mnist_loaders
from common.utils import configure_cpu, count_params, save_image_grid, set_seed
from a1_vae.model import VAE, elbo


def evaluate(model: VAE, loader, device) -> float:
    """Average test negative ELBO (nats/image)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            xb = dynamically_binarize(x)
            out = model(xb)
            neg_elbo, _, _ = elbo(out, xb)
            total += neg_elbo.item() * x.size(0)
            n += x.size(0)
    return total / n


@torch.no_grad()
def iwae_log_likelihood(model: VAE, loader, device, k: int = 50, max_batches: int = 20) -> float:
    """Importance-weighted estimate of E[log p(x)] (Burda et al. 2016, IWAE).

    log p(x) >= log (1/k) sum_i w_i,  w_i = p(x|z_i) p(z_i) / q(z_i|x), z_i ~ q(z|x).
    A tighter bound than the ELBO; reported as nats/image (higher = better).
    """
    model.eval()
    total, n, batches = 0.0, 0, 0
    for x, _ in loader:
        x = x.to(device)
        xb = dynamically_binarize(x).view(x.size(0), -1)
        b = xb.size(0)
        mu, logvar = model.encode(xb)
        std = torch.exp(0.5 * logvar)
        # (k, b, D)
        eps = torch.randn(k, b, model.latent, device=device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        logits = model.decode(z)  # (k, b, 784)
        x_rep = xb.unsqueeze(0).expand(k, b, -1)
        log_px_z = -F.binary_cross_entropy_with_logits(logits, x_rep, reduction="none").sum(-1)
        log_pz = (-0.5 * (z ** 2 + math.log(2 * math.pi))).sum(-1)
        log_qz = (-0.5 * (eps ** 2 + math.log(2 * math.pi) + logvar.unsqueeze(0))).sum(-1)
        log_w = log_px_z + log_pz - log_qz  # (k, b)
        log_px = torch.logsumexp(log_w, dim=0) - math.log(k)  # (b,)
        total += log_px.sum().item()
        n += b
        batches += 1
        if batches >= max_batches:
            break
    return total / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=20)
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--train-subset", type=int, default=None, help="limit train set for speed")
    p.add_argument("--results", type=str, default="results/a1_vae")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    configure_cpu()
    device = torch.device("cpu")
    os.makedirs(args.results, exist_ok=True)

    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size, train_subset=args.train_subset
    )
    model = VAE(hidden=args.hidden, latent=args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"VAE params: {count_params(model):,}")

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            xb = dynamically_binarize(x)
            out = model(xb)
            neg_elbo, recon, kl = elbo(out, xb)
            opt.zero_grad()
            neg_elbo.backward()
            opt.step()
            run += neg_elbo.item()
            nb += 1
        train_nelbo = run / nb
        test_nelbo = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_neg_elbo": train_nelbo, "test_neg_elbo": test_nelbo})
        print(f"epoch {epoch:2d}  train -ELBO {train_nelbo:8.3f}  test -ELBO {test_nelbo:8.3f}")

    train_time = time.time() - t0

    # Evidence: samples from the prior + reconstructions.
    samples = model.sample(64, device=device)
    save_image_grid(samples, os.path.join(args.results, "samples.png"), nrow=8)

    x, _ = next(iter(test_loader))
    x = x[:8].to(device)
    xb = dynamically_binarize(x)
    with torch.no_grad():
        recon = torch.sigmoid(model(xb).logits).view(-1, 1, 28, 28)
    pair = torch.cat([xb.view(-1, 1, 28, 28)[:8], recon[:8]], dim=0)
    save_image_grid(pair, os.path.join(args.results, "reconstructions.png"), nrow=8)

    iwae = iwae_log_likelihood(model, test_loader, device, k=50)
    metrics = {
        "model": "VAE (MLP, Kingma & Welling 2014)",
        "params": count_params(model),
        "epochs": args.epochs,
        "latent_dim": args.latent,
        "final_test_neg_elbo_nats": history[-1]["test_neg_elbo"],
        "iwae_loglik_nats_k50": iwae,
        "train_seconds": round(train_time, 1),
        "history": history,
    }
    with open(os.path.join(args.results, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTest -ELBO: {metrics['final_test_neg_elbo_nats']:.3f} nats")
    print(f"IWAE log p(x) (k=50): {iwae:.3f} nats")
    print(f"Saved samples + metrics to {args.results}")


if __name__ == "__main__":
    main()
