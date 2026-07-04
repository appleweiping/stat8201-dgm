"""Train PixelCNN on dynamically binarized MNIST; report NLL (nats & bits/dim) + samples.

Run:
    python -m a4_autoregressive.train --epochs 6 --results results/a4_pixelcnn

Produces:
    results/a4_pixelcnn/samples.png  -- ancestrally sampled digits
    results/a4_pixelcnn/metrics.json -- test negative log-likelihood
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
from a4_autoregressive.model import PixelCNN


def nll_nats(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Bernoulli NLL summed over the 784 pixels, averaged over the batch."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1).mean()


def evaluate(model: PixelCNN, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            xb = dynamically_binarize(x)
            logits = model(xb)
            loss = nll_nats(logits, xb)
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-res", type=int, default=5)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--results", type=str, default="results/a4_pixelcnn")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    configure_cpu()
    device = torch.device("cpu")
    os.makedirs(args.results, exist_ok=True)

    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size, train_subset=args.train_subset
    )
    model = PixelCNN(hidden=args.hidden, n_res=args.n_res).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"PixelCNN params: {count_params(model):,}")

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            xb = dynamically_binarize(x)
            logits = model(xb)
            loss = nll_nats(logits, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        train_nll = run / nb
        test_nll = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_nll_nats": train_nll, "test_nll_nats": test_nll})
        bpd = test_nll / (784 * math.log(2))
        print(f"epoch {epoch:2d}  train NLL {train_nll:8.3f}  test NLL {test_nll:8.3f}  ({bpd:.4f} bpd)")

    train_time = time.time() - t0

    print("sampling (sequential, may take ~1-2 min)...")
    samples = model.sample(64, device=device)
    save_image_grid(samples, os.path.join(args.results, "samples.png"), nrow=8)

    test_nll = history[-1]["test_nll_nats"]
    metrics = {
        "model": "PixelCNN (van den Oord et al. 2016)",
        "params": count_params(model),
        "epochs": args.epochs,
        "final_test_nll_nats": test_nll,
        "final_test_bits_per_dim": test_nll / (784 * math.log(2)),
        "train_seconds": round(train_time, 1),
        "history": history,
    }
    with open(os.path.join(args.results, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTest NLL: {test_nll:.3f} nats ({metrics['final_test_bits_per_dim']:.4f} bpd)")
    print(f"Saved samples + metrics to {args.results}")


if __name__ == "__main__":
    main()
