"""
scripts/eval_kid.py  —  Part 6B: KID evaluation
=================================================
Compute KID (Kernel Inception Distance) for each method and step count
to fill in the table in Problem 6.B.

Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \\
        --vp_checkpoint  runs/vp/best.pt \\
        --rf_checkpoint  runs/rectflow/best.pt \\
        --beta_min 0.01 --beta_max 5.0 \\
        --n_samples 1000 --device cuda

The script prints a markdown table with KID mean ± std for each
(method, num_steps) combination.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError(
        "torch-fidelity is required. Install with: pip install torch-fidelity"
    )

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    """Save (B,1,H,W) samples to individual PNG files for torch-fidelity."""
    os.makedirs(directory, exist_ok=True)
    samples = (samples.clamp(-1, 1) * 0.5 + 0.5)  # [0,1]
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def compute_kid(generated_dir: str, real_dir: str) -> dict:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics


def main():
    args = get_args()
    device = torch.device(args.device)

def main():
    args   = get_args()
    device = torch.device(args.device)

    # ── Load models ───────────────────────────────────────────────────────────
    sde       = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    vp_model  = UNet(in_channels=1, base_channels=64).to(device)
    vp_model.load_state_dict(torch.load(args.vp_checkpoint, map_location=device))
    vp_model.eval()

    flow      = RectifiedFlow()
    rf_model  = UNet(in_channels=1, base_channels=64).to(device)
    rf_model.load_state_dict(torch.load(args.rf_checkpoint, map_location=device))
    rf_model.eval()

    # ── Save real FashionMNIST test images once ───────────────────────────────
    real_dir = tempfile.mkdtemp(prefix="kid_real_")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    real_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    real_samples = torch.stack([real_ds[i][0] for i in range(args.n_samples)])
    save_samples_to_dir(real_samples, real_dir)
    print(f"Saved {args.n_samples} real images to {real_dir}")

    shape = (args.n_samples, 1, 28, 28)

    # ── Evaluate each (method, steps) combination ─────────────────────────────
    results = {}   # key: (method, steps) -> (mean, std)

    for steps in STEP_COUNTS:
        gen_dir = tempfile.mkdtemp(prefix=f"kid_gen_{steps}_")

        # Rectified Flow
        with torch.no_grad():
            samples = flow.euler_sample(rf_model, shape, num_steps=steps, device=device)
        save_samples_to_dir(samples.cpu(), gen_dir)
        m = compute_kid(gen_dir, real_dir)
        results[("rectflow", steps)] = (m["kernel_inception_distance_mean"],
                                        m["kernel_inception_distance_std"])

        # VP EM  (skip step counts that are too slow at 1k samples)
        gen_dir_em = tempfile.mkdtemp(prefix=f"kid_em_{steps}_")
        with torch.no_grad():
            samples_em = sde.euler_maruyama(vp_model, shape, num_steps=steps, device=device)
        save_samples_to_dir(samples_em.cpu(), gen_dir_em)
        m_em = compute_kid(gen_dir_em, real_dir)
        results[("em", steps)] = (m_em["kernel_inception_distance_mean"],
                                  m_em["kernel_inception_distance_std"])

        print(f"steps={steps:5d} | RF  KID = {results[('rectflow', steps)][0]:.4f} "
              f"± {results[('rectflow', steps)][1]:.4f} | "
              f"VP EM KID = {results[('em', steps)][0]:.4f} "
              f"± {results[('em', steps)][1]:.4f}")

    # ── Print markdown table ──────────────────────────────────────────────────
    print("\n| Steps | Flow Matching | DDPM EM |")
    print("|------:|:-------------|:--------|")
    for steps in STEP_COUNTS:
        rf_m, rf_s = results[("rectflow", steps)]
        em_m, em_s = results[("em",       steps)]
        baseline   = " ← baseline" if steps == 1000 else ""
        print(f"| {steps:5d} | {rf_m:.4f} ± {rf_s:.4f} | "
              f"{em_m:.4f} ± {em_s:.4f}{baseline} |")


if __name__ == "__main__":
    main()
