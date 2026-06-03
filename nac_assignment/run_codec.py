from __future__ import annotations

import argparse
import os
from pathlib import Path

import dac
import torch
import torchaudio
from tqdm import tqdm

from common import (
    TARGET_SAMPLE_RATE,
    load_audio,
    manifest_path,
    models_dir,
    outputs_dir,
    read_jsonl,
    reconstructed_path,
    write_jsonl,
)

RVQ_SETTINGS = (
    ("full_rvq", None),
    ("rvq_5", 5),
    ("rvq_3", 3),
    ("rvq_2", 2),
    ("rvq_1", 1),
)


def download_dac_model() -> Path:
    local_home = models_dir() / "dac_home"
    local_home.mkdir(parents=True, exist_ok=True)
    previous_home = os.environ.get("HOME")
    os.environ["HOME"] = str(local_home)
    try:
        model_path = Path(dac.utils.download(model_type="24khz"))
    finally:
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home
    if local_home.resolve() not in model_path.resolve().parents:
        raise RuntimeError(
            "DAC package did not honor the workspace-local model download directory"
        )
    return model_path


class LogMelDifference:
    def __init__(self, device: torch.device) -> None:
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            f_min=0,
            f_max=12_000,
            power=2.0,
        ).to(device)

    def __call__(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        original_mel = self.transform(original)
        reconstructed_mel = self.transform(reconstructed)
        frames = min(original_mel.shape[-1], reconstructed_mel.shape[-1])
        original_log_mel = torch.log(original_mel[..., :frames].clamp_min(1e-5))
        reconstructed_log_mel = torch.log(
            reconstructed_mel[..., :frames].clamp_min(1e-5)
        )
        return (original_log_mel - reconstructed_log_mel).abs().mean().item()


def process_dataset(
    dataset: str,
    model: dac.DAC,
    device: torch.device,
    force: bool,
) -> None:
    rows = read_jsonl(manifest_path(dataset))
    if not rows:
        raise RuntimeError(f"Dataset manifest is missing or empty: {dataset}")
    mel_difference = LogMelDifference(device)
    metric_rows = []
    for row in tqdm(rows, desc=f"DAC {dataset}"):
        original = load_audio(Path(row["audio_path"])).to(device)
        original_destination = reconstructed_path(dataset, "original", row["id"])
        original_destination.parent.mkdir(parents=True, exist_ok=True)
        if force or not original_destination.exists():
            torchaudio.save(original_destination, original.cpu(), TARGET_SAMPLE_RATE)
        batched = original.unsqueeze(0)
        for condition, rvq_levels in RVQ_SETTINGS:
            destination = reconstructed_path(dataset, condition, row["id"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if force or not destination.exists():
                with torch.inference_mode():
                    output = model(batched, sample_rate=TARGET_SAMPLE_RATE, n_quantizers=rvq_levels)
                    reconstructed = output["audio"][0].clamp(-1.0, 1.0)
                torchaudio.save(destination, reconstructed.cpu(), TARGET_SAMPLE_RATE)
            else:
                reconstructed = load_audio(destination).to(device)
            metric_rows.append(
                {
                    "id": row["id"],
                    "dataset": dataset,
                    "condition": condition,
                    "rvq_levels": model.n_codebooks if rvq_levels is None else rvq_levels,
                    "logmel_diff": mel_difference(original, reconstructed),
                }
            )
    metric_path = outputs_dir() / "metrics" / "codec" / f"{dataset}.jsonl"
    write_jsonl(metric_path, metric_rows)
    print(f"[DONE] Codec metrics: {metric_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ljspeech", "vctk"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model_path = download_dac_model()
    print(f"[INFO] Loading DAC 24kHz model from {model_path}")
    model = dac.DAC.load(model_path).to(device).eval()
    print(f"[INFO] DAC full RVQ levels: {model.n_codebooks}")
    for dataset in args.datasets:
        process_dataset(dataset, model, device, args.force)


if __name__ == "__main__":
    main()
