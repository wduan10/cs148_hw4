"""
scripts/sample.py  —  Generate and compare samples (Parts 5C, 6B, 6D)
=======================================================================

Usage::
    # EM samples  (5.C.iii)
    python scripts/sample.py --method em --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000

    # PC samples  (5.C.iv)
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 1
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 3

    # Rectified Flow Euler  (6.B)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow/best.pt \\
        --num_steps 100

    # One-step reflow  (6.C)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow_reflow/best.pt \\
        --num_steps 1

    # Side-by-side grid  (6.D): pass a fixed seed file
    python scripts/sample.py --method all --vp_checkpoint runs/vp/best.pt \\
        --rf_checkpoint runs/rectflow/best.pt \\
        --reflow_checkpoint runs/rectflow_reflow/best.pt \\
        --seed 42 --out comparison_grid.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def save_grid(samples: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a (B,1,H,W) tensor as an image grid."""
    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=nrow)
    plt.figure(figsize=(nrow, samples.size(0) // nrow + 1))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",      type=str, default="em",
                   choices=["em", "pc", "rectflow", "all"],
                   help="Sampler to run (or 'all' for side-by-side grid).")
    # VP checkpoints
    p.add_argument("--checkpoint",    type=str, default=None)
    p.add_argument("--vp_checkpoint", type=str, default=None)
    # Rect-flow checkpoints
    p.add_argument("--rf_checkpoint",     type=str, default=None)
    p.add_argument("--reflow_checkpoint", type=str, default=None)
    # VP schedule
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--T",        type=int,   default=1000)
    # Sampler params
    p.add_argument("--num_steps",   type=int, default=1000)
    p.add_argument("--n_corrector", type=int, default=1)
    p.add_argument("--snr",         type=float, default=0.16)
    p.add_argument("--n_samples",   type=int, default=64)
    # Output
    p.add_argument("--out",    type=str, default="samples.png")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vp_model(checkpoint: str, device, beta_min=0.01, beta_max=5.0, T=1000):
    sde   = VPSDE(beta_min=beta_min, beta_max=beta_max, T=T)
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return sde, model


def load_rf_model(checkpoint: str, device):
    flow  = RectifiedFlow()
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return flow, model


# ── Helpers that accept a pre-generated initial state ─────────────────────────

@torch.no_grad()
def em_from_init(sde: VPSDE, model, x_init: torch.Tensor, num_steps: int, device) -> torch.Tensor:
    """VP Euler-Maruyama reverse SDE from a given initial x."""
    dt = 1.0 / num_steps
    timesteps = torch.linspace(1.0, dt, num_steps, device=device)
    B    = x_init.shape[0]
    ndim = x_init.dim() - 1
    x    = x_init.clone().to(device)
    for t_val in timesteps:
        t      = torch.full((B,), t_val.item(), device=device)
        score  = model(x, t)
        beta_t = sde.beta(t).view(B, *([1] * ndim))
        drift  = (0.5 * beta_t * x + beta_t * score) * dt
        diff   = torch.sqrt(beta_t * dt) * torch.randn_like(x)
        x      = x + drift + diff
    return x


@torch.no_grad()
def rf_from_init(model, x_init: torch.Tensor, num_steps: int, device) -> torch.Tensor:
    """Rectified Flow Euler ODE from a given initial x."""
    dt = 1.0 / num_steps
    B  = x_init.shape[0]
    x  = x_init.clone().to(device)
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        x = x + model(x, t) * dt
    return x


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape  = (args.n_samples, 1, 28, 28)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.method == "em":
        sde, model = load_vp_model(
            args.checkpoint, device, args.beta_min, args.beta_max, args.T)
        samples = sde.euler_maruyama(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out,
                  title=f"VP Euler-Maruyama ({args.num_steps} steps)")

    elif args.method == "pc":
        sde, model = load_vp_model(
            args.checkpoint, device, args.beta_min, args.beta_max, args.T)
        samples = sde.predictor_corrector(
            model, shape,
            num_steps=args.num_steps,
            n_corrector=args.n_corrector,
            snr=args.snr,
            device=device,
        )
        save_grid(samples, args.out,
                  title=f"VP PC ({args.num_steps} steps, {args.n_corrector} correctors)")

    elif args.method == "rectflow":
        flow, model = load_rf_model(args.checkpoint, device)
        samples = flow.euler_sample(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out,
                  title=f"Rectified Flow Euler ({args.num_steps} steps)")

    elif args.method == "all":
        # ── 6.D: 4×8 side-by-side grid with 8 fixed seeds ────────────────
        n_seeds = 8
        torch.manual_seed(args.seed)
        base_noise = torch.randn(n_seeds, 1, 28, 28, device=device)  # shared base noise

        rows = []
        row_labels = []

        # Row 1: DDPM EM (1000 steps)
        sde, vp_model = load_vp_model(
            args.vp_checkpoint, device, args.beta_min, args.beta_max, args.T)
        t1     = torch.ones(n_seeds, device=device)
        sigma1 = sde.sigma(t1).view(n_seeds, 1, 1, 1)
        rows.append(em_from_init(sde, vp_model, base_noise * sigma1, 1000, device))
        row_labels.append("DDPM EM (1000 steps)")

        # Row 2: Rectified Flow (100 steps)
        flow, rf_model = load_rf_model(args.rf_checkpoint, device)
        rows.append(rf_from_init(rf_model, base_noise, 100, device))
        row_labels.append("Rect. Flow (100 steps)")

        # Row 3: Rectified Flow (1 step)
        rows.append(rf_from_init(rf_model, base_noise, 1, device))
        row_labels.append("Rect. Flow (1 step)")

        # Row 4: Reflow (1 step)
        flow2, reflow_model = load_rf_model(args.reflow_checkpoint, device)
        rows.append(rf_from_init(reflow_model, base_noise, 1, device))
        row_labels.append("Reflow (1 step)")

        all_samples = torch.cat(rows, dim=0)   # (32, 1, 28, 28)
        save_grid(all_samples, args.out, nrow=n_seeds,
                  title=" | ".join(row_labels))
        print("Row order:", row_labels)


if __name__ == "__main__":
    main()
