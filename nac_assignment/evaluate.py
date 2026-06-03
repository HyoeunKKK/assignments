from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from jiwer import wer
from tqdm import tqdm

from common import (
    CONDITIONS,
    append_jsonl,
    condition_audio_path,
    manifest_path,
    models_dir,
    outputs_dir,
    read_jsonl,
    write_jsonl,
)

WHISPER_BEAM_SIZE = 1
UTMOS_BATCH_SIZE = 8
UTMOS_NUM_WORKERS = 0


def metric_path(metric: str, dataset: str, condition: str) -> Path:
    return outputs_dir() / "metrics" / metric / dataset / f"{condition}.jsonl"


def metric_rows_by_id(metric: str, dataset: str, condition: str) -> dict[str, dict]:
    return {
        row["id"]: row for row in read_jsonl(metric_path(metric, dataset, condition))
    }


def evaluate_wer(
    datasets: list[str],
    device: str,
    whisper_model: str,
    batch_size: int,
) -> None:
    import whisper
    from whisper.normalizers import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()
    whisper_root = models_dir() / "whisper"
    whisper_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading Whisper {whisper_model}")
    model = whisper.load_model(whisper_model, device=device, download_root=whisper_root)
    options = whisper.DecodingOptions(
        language="en",
        task="transcribe",
        beam_size=WHISPER_BEAM_SIZE,
        fp16=device.startswith("cuda"),
        without_timestamps=True,
    )
    for dataset in datasets:
        manifest_rows = read_jsonl(manifest_path(dataset))
        for condition in CONDITIONS:
            destination = metric_path("wer", dataset, condition)
            completed = metric_rows_by_id("wer", dataset, condition)
            pending_rows = [row for row in manifest_rows if row["id"] not in completed]
            for start in tqdm(
                range(0, len(pending_rows), batch_size),
                desc=f"WER {dataset}/{condition}",
            ):
                batch_rows = pending_rows[start : start + batch_size]
                mels = []
                for row in batch_rows:
                    audio_path = condition_audio_path(dataset, condition, row)
                    audio = whisper.load_audio(str(audio_path))
                    audio = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(
                        audio, n_mels=model.dims.n_mels
                    ).to(model.device)
                    mels.append(mel)
                results = model.decode(
                    torch.stack(mels),
                    options,
                )
                for row, result in zip(batch_rows, results):
                    reference = row["text"]
                    hypothesis = result.text
                    append_jsonl(
                        destination,
                        {
                            "id": row["id"],
                            "dataset": dataset,
                            "condition": condition,
                            "reference": reference,
                            "hypothesis": hypothesis,
                            "normalized_reference": normalizer(reference),
                            "normalized_hypothesis": normalizer(hypothesis),
                            "whisper_model": whisper_model,
                            "beam_size": WHISPER_BEAM_SIZE,
                            "batch_size": batch_size,
                        },
                    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_utmos(datasets: list[str], device: str) -> None:
    os.environ.setdefault("UTMOSV2_CHACHE", str(models_dir() / "utmosv2_cache"))
    import utmosv2

    print("[INFO] Loading UTMOSv2")
    model = utmosv2.create_model(pretrained=True, device=device)
    for dataset in datasets:
        manifest_rows = read_jsonl(manifest_path(dataset))
        expected_ids = {row["id"] for row in manifest_rows}
        for condition in CONDITIONS:
            destination = metric_path("utmos", dataset, condition)
            completed = metric_rows_by_id("utmos", dataset, condition)
            if set(completed) == expected_ids:
                continue
            input_dir = outputs_dir() / "audio" / dataset / condition
            predictions = model.predict(
                input_dir=str(input_dir),
                device=device,
                batch_size=UTMOS_BATCH_SIZE,
                num_workers=UTMOS_NUM_WORKERS,
                verbose=True,
            )
            rows = [
                {
                    "id": Path(prediction["file_path"]).stem,
                    "dataset": dataset,
                    "condition": condition,
                    "utmos": float(prediction["predicted_mos"]),
                    "batch_size": UTMOS_BATCH_SIZE,
                }
                for prediction in predictions
            ]
            predicted_ids = {row["id"] for row in rows}
            if predicted_ids != expected_ids:
                raise RuntimeError(
                    f"UTMOS inputs do not match {dataset}/{condition}: "
                    f"expected {len(expected_ids)}, found {len(predicted_ids)}"
                )
            write_jsonl(
                destination,
                rows,
            )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def require_complete_rows(
    metric: str,
    dataset: str,
    condition: str,
    rows: list[dict],
    expected_ids: set[str],
) -> None:
    row_ids = [row["id"] for row in rows]
    if len(row_ids) != len(expected_ids) or set(row_ids) != expected_ids:
        raise RuntimeError(
            f"{metric} results incomplete for {dataset}/{condition}: "
            f"expected {len(expected_ids)} unique IDs, found "
            f"{len(row_ids)} rows and {len(set(row_ids))} unique IDs"
        )


def codec_means(dataset: str, expected_ids: set[str]) -> dict[str, float]:
    rows = read_jsonl(outputs_dir() / "metrics" / "codec" / f"{dataset}.jsonl")
    values: dict[str, list[float]] = defaultdict(list)
    for condition in CONDITIONS[1:]:
        condition_rows = [row for row in rows if row["condition"] == condition]
        require_complete_rows("codec", dataset, condition, condition_rows, expected_ids)
        values[condition] = [row["logmel_diff"] for row in condition_rows]
    return {
        condition: mean(condition_values)
        for condition, condition_values in values.items()
    }


def corpus_wer(dataset: str, condition: str, expected_ids: set[str]) -> float:
    rows = read_jsonl(metric_path("wer", dataset, condition))
    require_complete_rows("WER", dataset, condition, rows, expected_ids)
    return wer(
        [row["normalized_reference"] for row in rows],
        [row["normalized_hypothesis"] for row in rows],
    )


def mean_utmos(dataset: str, condition: str, expected_ids: set[str]) -> float:
    rows = read_jsonl(metric_path("utmos", dataset, condition))
    require_complete_rows("UTMOS", dataset, condition, rows, expected_ids)
    return mean(row["utmos"] for row in rows)


def aggregate(datasets: list[str]) -> None:
    summary = {}
    summary_dir = outputs_dir() / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        expected_ids = {row["id"] for row in read_jsonl(manifest_path(dataset))}
        if not expected_ids:
            raise RuntimeError(f"Dataset manifest is missing or empty: {dataset}")
        mel = codec_means(dataset, expected_ids)
        dataset_rows = []
        for condition in CONDITIONS:
            dataset_rows.append(
                {
                    "condition": condition,
                    "logmel_diff": None if condition == "original" else mel[condition],
                    "wer": corpus_wer(dataset, condition, expected_ids),
                    "utmos": mean_utmos(dataset, condition, expected_ids),
                }
            )
        summary[dataset] = dataset_rows
        csv_path = summary_dir / f"{dataset}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=dataset_rows[0].keys())
            writer.writeheader()
            writer.writerows(dataset_rows)
        print(f"[DONE] Summary table: {csv_path}")

    (summary_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    markdown = []
    for dataset, rows in summary.items():
        markdown.extend(
            [
                f"## {dataset}",
                "",
                "| Setup | LogMel diff. (L1) | WER | UTMOS |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            logmel = "-" if row["logmel_diff"] is None else f"{row['logmel_diff']:.4f}"
            markdown.append(
                f"| {row['condition']} | {logmel} | {row['wer']:.4f} | {row['utmos']:.4f} |"
            )
        markdown.append("")
    (summary_dir / "tables.md").write_text("\n".join(markdown), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ljspeech", "vctk"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--metric", choices=["wer", "utmos", "aggregate", "all"], default="all"
    )
    parser.add_argument("--whisper-model", default="large")
    parser.add_argument("--whisper-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metric in ("wer", "all"):
        evaluate_wer(
            args.datasets,
            args.device,
            args.whisper_model,
            args.whisper_batch_size,
        )
    if args.metric in ("utmos", "all"):
        evaluate_utmos(args.datasets, args.device)
    if args.metric in ("aggregate", "all"):
        aggregate(args.datasets)


if __name__ == "__main__":
    main()
