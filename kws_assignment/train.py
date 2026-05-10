import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import GSCDataset
from src.models.dscnn import DSCNN, count_parameters

import wandb

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bsz = x.size(0)
        loss_sum += loss.item() * bsz
        acc_sum += accuracy(logits, y) * bsz
        n += bsz

    return loss_sum / n, acc_sum / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dscnn.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds = GSCDataset(
        csv_path=cfg["data"]["train_csv"],
        sample_rate=cfg["data"]["sample_rate"],
        clip_samples=cfg["data"]["clip_samples"],
        n_fft=cfg["feature"]["n_fft"],
        win_length=cfg["feature"]["win_length"],
        hop_length=cfg["feature"]["hop_length"],
        n_mels=cfg["feature"]["n_mels"],
        train=True,
    )
    valid_ds = GSCDataset(
        csv_path=cfg["data"]["valid_csv"],
        sample_rate=cfg["data"]["sample_rate"],
        clip_samples=cfg["data"]["clip_samples"],
        n_fft=cfg["feature"]["n_fft"],
        win_length=cfg["feature"]["win_length"],
        hop_length=cfg["feature"]["hop_length"],
        n_mels=cfg["feature"]["n_mels"],
        train=False,
    )
    test_ds = GSCDataset(
        csv_path=cfg["data"]["test_csv"],
        sample_rate=cfg["data"]["sample_rate"],
        clip_samples=cfg["data"]["clip_samples"],
        n_fft=cfg["feature"]["n_fft"],
        win_length=cfg["feature"]["win_length"],
        hop_length=cfg["feature"]["hop_length"],
        n_mels=cfg["feature"]["n_mels"],
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model = DSCNN(
        num_classes=cfg["model"]["num_classes"],
        channels=cfg["model"]["channels"],
        block_strides=cfg["model"]["block_strides"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    n_params = count_parameters(model)
    print(f"trainable params: {n_params:,}")
    assert n_params <= 2_500_000, "Model exceeds 2.5M parameters"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    os.makedirs(os.path.dirname(cfg["train"]["ckpt_path"]), exist_ok=True)
    writer = SummaryWriter(cfg["train"]["log_dir"])

    run = wandb.init(
        project="kws_assignment",
        name="dscnn_baseline",
        config=cfg,
    )
    wandb.watch(model, log="all", log_freq=100)

    best_valid_acc = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        grad_norm_sum = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            gnorm = grad_norm(model)
            optimizer.step()

            bsz = x.size(0)
            train_loss_sum += loss.item() * bsz
            train_acc_sum += accuracy(logits, y) * bsz
            grad_norm_sum += gnorm * bsz
            n += bsz

        scheduler.step()

        train_loss = train_loss_sum / n
        train_acc = train_acc_sum / n
        avg_grad_norm = grad_norm_sum / n

        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("valid/loss", valid_loss, epoch)
        writer.add_scalar("valid/acc", valid_acc, epoch)
        writer.add_scalar("train/lr", lr, epoch)
        writer.add_scalar("train/grad_norm", avg_grad_norm, epoch)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "valid/loss": valid_loss,
                "valid/acc": valid_acc,
                "train/lr": lr,
                "train/grad_norm": avg_grad_norm,
            },
            step=epoch,
        )

        print(
            f"[{epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_acc={valid_acc:.4f} "
            f"lr={lr:.6f} "
            f"grad_norm={avg_grad_norm:.4f}"
        )

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "best_valid_acc": best_valid_acc,
                },
                cfg["train"]["ckpt_path"],
            )

    print(f"best valid acc: {best_valid_acc:.4f}")

    ckpt = torch.load(cfg["train"]["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    writer.add_scalar("test/acc", test_acc, 0)
    writer.add_scalar("test/loss", test_loss, 0)

    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    wandb.log(
        {
            "test/loss": test_loss,
            "test/acc": test_acc,
        }
    )
    run.finish()
    writer.close()


if __name__ == "__main__":
    main()