"""Train a DDPM on MNIST (eps-prediction) and sample digits from pure noise.

Run:
    python -m a5_diffusion.train --epochs 12 --results results/a5_diffusion

Produces:
    results/a5_diffusion/samples.png  -- digits sampled by running the reverse chain
    results/a5_diffusion/metrics.json -- final training loss + config
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from common.data import get_mnist_loaders
from common.utils import configure_cpu, count_params, save_image_grid, set_seed
from a5_diffusion.model import DDPM, UNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--results", type=str, default="results/a5_diffusion")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    configure_cpu()
    device = torch.device("cpu")
    os.makedirs(args.results, exist_ok=True)

    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size, train_subset=args.train_subset)
    net = UNet(base=args.base).to(device)
    ddpm = DDPM(net, T=args.timesteps).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"DDPM U-Net params: {count_params(net):,}  T={args.timesteps}")

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        run, nb = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device) * 2 - 1  # to [-1, 1]
            loss = ddpm.loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        train_loss = run / nb
        history.append({"epoch": epoch, "train_mse": train_loss})
        print(f"epoch {epoch:2d}  train MSE {train_loss:.5f}")

    train_time = time.time() - t0

    print("sampling reverse chain...")
    samples = ddpm.sample(64, device=device)
    save_image_grid(samples, os.path.join(args.results, "samples.png"), nrow=8)

    metrics = {
        "model": "DDPM (Ho et al. 2020), U-Net eps-predictor",
        "params": count_params(net),
        "timesteps": args.timesteps,
        "epochs": args.epochs,
        "final_train_mse": history[-1]["train_mse"],
        "train_seconds": round(train_time, 1),
        "history": history,
    }
    with open(os.path.join(args.results, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFinal train MSE: {metrics['final_train_mse']:.5f}")
    print(f"Saved samples + metrics to {args.results}")


if __name__ == "__main__":
    main()
