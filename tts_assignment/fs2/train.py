"""Train FastSpeech2 on LJSpeech."""

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .dataset import LJSpeechDataset, collate_fn
from .model import FastSpeech2, make_pad_mask
from .text import PAD_ID, VOCAB_SIZE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NoamScheduler:
    def __init__(self, optimizer, d_model: int, warmup: int = 4000, scale: float = 1.0):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.scale = scale
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.scale * (self.d_model ** -0.5) * min(self.step_num ** -0.5,
                                                       self.step_num * (self.warmup ** -1.5))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def compute_losses(out: dict, batch: dict):
    """Return loss dict.

    mel_before / mel_after L1 loss (only over real mel frames)
    duration loss: MSE between predicted log-duration and log(gt+1) on real phonemes.
    """
    mel_target = batch["mels"].transpose(1, 2)  # (B, T, n_mels)
    mel_lengths = batch["mel_lengths"]
    tgt_mask = make_pad_mask(mel_lengths, max_len=mel_target.size(1))
    valid = (~tgt_mask).unsqueeze(-1).float()
    # Align lengths between target and prediction (they should match in training due to teacher forcing,
    # but in case of off-by-one mismatch, trim to min).
    T_pred = out["mel_after"].size(1)
    T_tgt = mel_target.size(1)
    T = min(T_pred, T_tgt)
    mel_target = mel_target[:, :T]
    mel_before = out["mel_before"][:, :T]
    mel_after = out["mel_after"][:, :T]
    valid = valid[:, :T]
    n_valid = valid.sum().clamp(min=1.0)
    mel_loss_before = (F.l1_loss(mel_before, mel_target, reduction="none") * valid).sum() / (n_valid * mel_target.size(-1))
    mel_loss_after = (F.l1_loss(mel_after, mel_target, reduction="none") * valid).sum() / (n_valid * mel_target.size(-1))

    src_mask = out["src_mask"]
    log_dur_pred = out["log_duration_pred"]
    durations = batch["durations"]
    log_dur_tgt = torch.log(durations.float() + 1.0)
    src_valid = (~src_mask).float()
    n_src_valid = src_valid.sum().clamp(min=1.0)
    dur_loss = ((log_dur_pred - log_dur_tgt) ** 2 * src_valid).sum() / n_src_valid

    total = mel_loss_before + mel_loss_after + dur_loss
    return {
        "loss": total,
        "mel_before": mel_loss_before.detach(),
        "mel_after": mel_loss_after.detach(),
        "duration": dur_loss.detach(),
    }


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    sums = {"loss": 0.0, "mel_before": 0.0, "mel_after": 0.0, "duration": 0.0}
    n_batches = 0
    for batch in val_loader:
        batch = move_to_device(batch, device)
        out = model(batch["phonemes"], batch["phoneme_lengths"],
                    durations=batch["durations"], mel_lengths=batch["mel_lengths"])
        losses = compute_losses(out, batch)
        for k in sums:
            sums[k] += float(losses[k] if k == "loss" else losses[k])
        n_batches += 1
    model.train()
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/home/elicer/SLP_assignment03_LJSpeech_manifest/manifests/ljspeech_ipa_compressed.jsonl")
    p.add_argument("--splits_dir", default="/home/elicer/SLP_assignment03_LJSpeech_manifest/manifests/splits")
    p.add_argument("--wav_root", default="/home/elicer/project/data/LJSpeech-1.1/wavs")
    p.add_argument("--cache_dir", default="/home/elicer/project/cache/mels")
    p.add_argument("--out_dir", default="/home/elicer/project/runs/fs2")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr_scale", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--val_interval_steps", type=int, default=1000)
    p.add_argument("--ckpt_interval_steps", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=80000)
    p.add_argument("--wandb_project", default="slp-fs2-ljspeech")
    p.add_argument("--wandb_run", default="fs2-default")
    p.add_argument("--wandb_mode", default="online")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--resume", default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LJSpeechDataset(args.manifest, os.path.join(args.splits_dir, "train_ids.txt"),
                               args.wav_root, cache_dir=args.cache_dir)
    val_ds = LJSpeechDataset(args.manifest, os.path.join(args.splits_dir, "valid_ids.txt"),
                             args.wav_root, cache_dir=args.cache_dir)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_fn, pin_memory=True)

    model = FastSpeech2(vocab_size=VOCAB_SIZE,
                        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
                        enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                        pad_id=PAD_ID).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9,
                                  weight_decay=1e-6)
    scheduler = NoamScheduler(optimizer, args.d_model, warmup=args.warmup, scale=args.lr_scale)

    global_step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.step_num = ckpt.get("step", 0)
        global_step = scheduler.step_num
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at step {global_step}")

    wandb.init(project=args.wandb_project, name=args.wandb_run, dir=str(out_dir),
               mode=args.wandb_mode,
               config={**vars(args), "n_params": n_params})

    model.train()
    t_start = time.time()
    last_log_t = time.time()
    pbar_steps = 0
    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["phonemes"], batch["phoneme_lengths"],
                        durations=batch["durations"], mel_lengths=batch["mel_lengths"])
            losses = compute_losses(out, batch)
            loss = losses["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = scheduler.step()
            optimizer.step()
            global_step += 1
            pbar_steps += 1

            if global_step % args.log_interval == 0:
                dt = time.time() - last_log_t
                steps_per_sec = args.log_interval / max(dt, 1e-6)
                last_log_t = time.time()
                wandb.log({
                    "train/loss": float(loss.detach()),
                    "train/mel_before": float(losses["mel_before"]),
                    "train/mel_after": float(losses["mel_after"]),
                    "train/duration": float(losses["duration"]),
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/steps_per_sec": steps_per_sec,
                    "train/epoch": epoch,
                }, step=global_step)
                print(f"[step {global_step}] loss={float(loss):.4f} mel_a={float(losses['mel_after']):.4f} "
                      f"dur={float(losses['duration']):.4f} lr={lr:.2e} gn={float(grad_norm):.3f} "
                      f"sps={steps_per_sec:.2f}")

            if global_step % args.val_interval_steps == 0:
                vstats = validate(model, val_loader, device)
                wandb.log({f"val/{k}": v for k, v in vstats.items()}, step=global_step)
                print(f"[val step {global_step}] " + " ".join(f"{k}={v:.4f}" for k, v in vstats.items()))

            if global_step % args.ckpt_interval_steps == 0:
                ckpt_path = out_dir / f"step_{global_step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "epoch": epoch,
                    "args": vars(args),
                }, ckpt_path)
                # also save latest.pt
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "epoch": epoch,
                    "args": vars(args),
                }, out_dir / "latest.pt")
                print(f"saved {ckpt_path}")

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    # Final
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": global_step, "epoch": epoch, "args": vars(args)},
               out_dir / "latest.pt")
    wandb.finish()
    print(f"Done in {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
