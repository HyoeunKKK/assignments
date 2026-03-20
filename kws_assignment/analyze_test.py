import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import GSCDataset, LABELS
from src.models.dscnn import DSCNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dscnn.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reports/test_analysis")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

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

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model_kwargs = {
        "num_classes": cfg["model"]["num_classes"],
        "dropout": cfg["model"]["dropout"],
    }

    if "channels" in cfg["model"]:
        model_kwargs["channels"] = cfg["model"]["channels"]
    if "block_strides" in cfg["model"]:
        model_kwargs["block_strides"] = cfg["model"]["block_strides"]

    model = DSCNN(**model_kwargs).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_classes = len(LABELS)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1

    # 클래스별 accuracy (= recall)
    row_sums = cm.sum(axis=1)
    class_acc = np.divide(
        np.diag(cm),
        row_sums,
        out=np.zeros_like(np.diag(cm), dtype=float),
        where=row_sums != 0,
    )

    # precision
    col_sums = cm.sum(axis=0)
    class_prec = np.divide(
        np.diag(cm),
        col_sums,
        out=np.zeros_like(np.diag(cm), dtype=float),
        where=col_sums != 0,
    )

    # f1
    class_f1 = np.divide(
        2 * class_prec * class_acc,
        class_prec + class_acc,
        out=np.zeros_like(class_acc, dtype=float),
        where=(class_prec + class_acc) != 0,
    )

    overall_acc = np.trace(cm) / cm.sum()

    print(f"\nOverall test accuracy: {overall_acc:.4f}\n")
    print(f"{'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'count':>10}")
    print("-" * 55)
    for i, name in enumerate(LABELS):
        print(
            f"{name:<10} "
            f"{class_prec[i]:>10.4f} "
            f"{class_acc[i]:>10.4f} "
            f"{class_f1[i]:>10.4f} "
            f"{row_sums[i]:>10d}"
        )

    # 가장 많이 헷갈린 오분류 pair 출력
    print("\nTop confusing pairs:")
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)

    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and off_diag[i, j] > 0:
                pairs.append((off_diag[i, j], LABELS[i], LABELS[j]))

    pairs.sort(reverse=True)

    for count, true_name, pred_name in pairs[:15]:
        print(f"{true_name:>10} -> {pred_name:<10} : {count}")

    # confusion matrix 저장
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, LABELS, rotation=45, ha="right")
    plt.yticks(tick_marks, LABELS)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    save_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved confusion matrix to: {save_path}")


if __name__ == "__main__":
    main()