from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import torch
import torchaudio

TARGET_SAMPLE_RATE = 24_000
CONDITIONS = ("original", "full_rvq", "rvq_5", "rvq_3", "rvq_2", "rvq_1")


def workspace_dir() -> Path:
    default = Path(__file__).resolve().parent
    return Path(os.environ.get("A04_WORKSPACE", default)).resolve()


def data_dir() -> Path:
    return workspace_dir() / "data"


def models_dir() -> Path:
    return workspace_dir() / "models"


def outputs_dir() -> Path:
    return workspace_dir() / "outputs"


def manifest_path(dataset: str) -> Path:
    return data_dir() / "manifests" / f"{dataset}.jsonl"


def reconstructed_path(dataset: str, condition: str, utterance_id: str) -> Path:
    return outputs_dir() / "audio" / dataset / condition / f"{utterance_id}.wav"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_audio(path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    waveform, source_sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, source_sample_rate, sample_rate
        )
    return waveform.clamp(-1.0, 1.0)


def condition_audio_path(dataset: str, condition: str, row: dict) -> Path:
    if condition == "original":
        normalized_original = reconstructed_path(dataset, condition, row["id"])
        return normalized_original if normalized_original.exists() else Path(row["audio_path"])
    return reconstructed_path(dataset, condition, row["id"])
