"""Pull wandb history and produce summary plots used in the report."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hyoeunkim-sungkyunkwan-university/slp-fs2-ljspeech/q5bpag6e")
    p.add_argument("--out_dir", default="/home/elicer/project/report_assets")
    p.add_argument("--samples_per_curve", type=int, default=1000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    r = api.run(args.run)
    keys = [
        "train/loss", "train/mel_before", "train/mel_after", "train/duration",
        "train/lr", "train/grad_norm",
        "val/loss", "val/mel_after", "val/mel_before", "val/duration",
        "_step",
    ]
    print("fetching history...")
    hist = r.history(keys=keys, pandas=False, samples=args.samples_per_curve)

    # convert to per-key arrays of (step, value)
    series = {k: ([], []) for k in keys if k != "_step"}
    for row in hist:
        step = row.get("_step")
        if step is None:
            continue
        for k in series:
            v = row.get(k)
            if v is not None:
                series[k][0].append(step)
                series[k][1].append(v)

    # plot 1: train losses
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    ax = axes[0]
    for k, color in [("train/mel_after", "tab:blue"), ("val/mel_after", "tab:orange")]:
        xs, ys = series[k]
        ax.plot(xs, ys, label=k, linewidth=1.2, color=color)
    ax.set_xlabel("step"); ax.set_ylabel("mel L1"); ax.set_title("Mel L1 (after postnet)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    for k, color in [("train/duration", "tab:green"), ("val/duration", "tab:red")]:
        xs, ys = series[k]
        ax.plot(xs, ys, label=k, linewidth=1.2, color=color)
    ax.set_xlabel("step"); ax.set_ylabel("MSE on log-duration"); ax.set_title("Duration loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax = axes[2]
    xs, ys = series["train/lr"]
    ax.plot(xs, ys, label="lr", linewidth=1.2, color="tab:purple")
    ax2 = ax.twinx()
    xs2, ys2 = series["train/grad_norm"]
    ax2.plot(xs2, ys2, label="grad_norm", linewidth=1.2, color="tab:gray", alpha=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("lr (purple)"); ax2.set_ylabel("grad_norm (gray)")
    ax.set_title("LR & grad-norm"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "train_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    # plot 2: combined total loss
    fig, ax = plt.subplots(figsize=(6, 3))
    for k, color in [("train/loss", "tab:blue"), ("val/loss", "tab:orange")]:
        xs, ys = series[k]
        ax.plot(xs, ys, label=k, linewidth=1.2, color=color)
    ax.set_xlabel("step"); ax.set_ylabel("total loss"); ax.set_title("Total training/validation loss")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "loss_total.png", dpi=150)
    print(f"wrote {out_dir / 'loss_total.png'}")


if __name__ == "__main__":
    main()
