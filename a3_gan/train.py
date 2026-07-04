"""Train a DCGAN (BCE or WGAN-GP) on MNIST and save generated samples.

Run:
    python -m a3_gan.train --loss bce --epochs 8 --results results/a3_gan
    python -m a3_gan.train --loss wgan_gp --epochs 8 --results results/a3_gan_wgan

Produces:
    <results>/samples.png   -- generated digits from a fixed noise batch
    <results>/metrics.json  -- final generator/discriminator losses + config
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from common.data import get_mnist_loaders
from common.utils import configure_cpu, count_params, save_image_grid, set_seed
from a3_gan.model import Discriminator, Generator, weights_init


def gradient_penalty(critic: Discriminator, real: torch.Tensor, fake: torch.Tensor, device):
    """WGAN-GP penalty: E[(||grad_x_hat D(x_hat)|| - 1)^2] on interpolates (Gulrajani 2017)."""
    b = real.size(0)
    eps = torch.rand(b, 1, 1, 1, device=device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_hat = critic(x_hat)
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(b, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loss", choices=["bce", "wgan_gp"], default="bce")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--n-critic", type=int, default=5, help="critic steps per G step (WGAN)")
    p.add_argument("--gp-lambda", type=float, default=10.0)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--results", type=str, default="results/a3_gan")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    configure_cpu()
    device = torch.device("cpu")
    os.makedirs(args.results, exist_ok=True)

    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size, train_subset=args.train_subset)

    use_sigmoid = args.loss == "bce"
    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator(use_sigmoid=use_sigmoid).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    betas = (0.5, 0.999) if args.loss == "bce" else (0.0, 0.9)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=betas)
    print(f"G params: {count_params(G):,}  D params: {count_params(D):,}  loss={args.loss}")

    fixed_z = torch.randn(64, args.z_dim, device=device)
    n_critic = args.n_critic if args.loss == "wgan_gp" else 1

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        dloss_acc, gloss_acc, steps = 0.0, 0.0, 0
        data_iter = iter(train_loader)
        n_batches = len(train_loader)
        b = 0
        while b < n_batches:
            # --- train discriminator/critic n_critic times ---
            d_val = 0.0
            for _ in range(n_critic):
                try:
                    real, _ = next(data_iter)
                except StopIteration:
                    break
                b += 1
                real = real.to(device)
                bs = real.size(0)
                z = torch.randn(bs, args.z_dim, device=device)
                fake = G(z).detach()
                if args.loss == "bce":
                    d_real = D(real)
                    d_fake = D(fake)
                    loss_d = F.binary_cross_entropy(d_real, torch.ones_like(d_real)) + \
                        F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
                else:  # wgan_gp
                    d_real = D(real)
                    d_fake = D(fake)
                    gp = gradient_penalty(D, real, fake, device)
                    loss_d = d_fake.mean() - d_real.mean() + args.gp_lambda * gp
                optD.zero_grad()
                loss_d.backward()
                optD.step()
                d_val = loss_d.item()

            # --- train generator once ---
            z = torch.randn(args.batch_size, args.z_dim, device=device)
            fake = G(z)
            if args.loss == "bce":
                d_fake = D(fake)
                loss_g = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))  # non-saturating
            else:
                loss_g = -D(fake).mean()
            optG.zero_grad()
            loss_g.backward()
            optG.step()

            dloss_acc += d_val
            gloss_acc += loss_g.item()
            steps += 1

        rec = {
            "epoch": epoch,
            "loss_d": dloss_acc / max(steps, 1),
            "loss_g": gloss_acc / max(steps, 1),
        }
        history.append(rec)
        print(f"epoch {epoch:2d}  loss_D {rec['loss_d']:8.4f}  loss_G {rec['loss_g']:8.4f}")

    train_time = time.time() - t0

    G.eval()
    with torch.no_grad():
        samples = G(fixed_z)
    save_image_grid(samples, os.path.join(args.results, "samples.png"), nrow=8)

    metrics = {
        "model": f"DCGAN ({args.loss})",
        "loss": args.loss,
        "g_params": count_params(G),
        "d_params": count_params(D),
        "epochs": args.epochs,
        "final_loss_d": history[-1]["loss_d"],
        "final_loss_g": history[-1]["loss_g"],
        "train_seconds": round(train_time, 1),
        "history": history,
    }
    with open(os.path.join(args.results, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved samples + metrics to {args.results}")


if __name__ == "__main__":
    main()
