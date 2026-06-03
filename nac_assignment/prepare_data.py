from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tarfile
import time
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

import pyarrow.parquet as pq

from common import data_dir, manifest_path, write_jsonl

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
VCTK_HF_DATASET = "sanchit-gandhi/vctk"
VCTK_PARQUET_URL = (
    "https://huggingface.co/api/datasets/"
    f"{VCTK_HF_DATASET}/parquet/default/train/{{shard}}.parquet"
)
VCTK_SHARDS = (1, 7)
VCTK_SPEAKERS = ("p232", "p257")
LJSPEECH_ID = re.compile(r"LJ\d{3}-\d{4}")


def parse_ljspeech_metadata(data: bytes) -> list[dict]:
    rows = []
    for line in data.decode("utf-8").splitlines():
        utterance_id, text, normalized_text = line.split("|", maxsplit=2)
        rows.append(
            {
                "id": utterance_id,
                "text": normalized_text,
                "raw_text": text,
            }
        )
    return rows


def ids_from_manifest(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    ids = set(LJSPEECH_ID.findall(text))
    if not ids:
        raise ValueError(f"No LJSpeech IDs found in {path}")
    return ids


def select_ljspeech_ids(
    metadata: list[dict],
    validation_manifest: Path | None,
    validation_size: int,
    seed: int,
) -> tuple[set[str], dict]:
    if validation_manifest is not None:
        selected = ids_from_manifest(validation_manifest)
        source = {
            "mode": "provided_manifest",
            "manifest": str(validation_manifest.resolve()),
        }
    else:
        if validation_size > len(metadata):
            raise ValueError("validation_size exceeds the number of LJSpeech samples")
        rng = random.Random(seed)
        selected = {row["id"] for row in rng.sample(metadata, validation_size)}
        source = {
            "mode": "fallback_random_split",
            "seed": seed,
            "validation_size": validation_size,
        }
    known = {row["id"] for row in metadata}
    missing = sorted(selected - known)
    if missing:
        raise ValueError(f"Unknown LJSpeech IDs in validation manifest: {missing[:5]}")
    return selected, source


def prepare_ljspeech(
    validation_manifest: Path | None,
    validation_size: int,
    seed: int,
    force: bool,
) -> None:
    output_manifest = manifest_path("ljspeech")
    if output_manifest.exists() and not force:
        print(f"[SKIP] LJSpeech manifest already exists: {output_manifest}")
        return

    output_dir = data_dir() / "ljspeech"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict] | None = None
    selected_ids: set[str] | None = None
    split_source: dict | None = None
    extracted: set[str] = set()

    print(f"[INFO] Streaming LJSpeech from {LJSPEECH_URL}")
    with urllib.request.urlopen(LJSPEECH_URL) as response:
        with tarfile.open(fileobj=response, mode="r|bz2") as archive:
            for member in archive:
                if member.name.endswith("/metadata.csv"):
                    extracted_file = archive.extractfile(member)
                    if extracted_file is None:
                        raise RuntimeError("Could not read LJSpeech metadata.csv")
                    metadata = parse_ljspeech_metadata(extracted_file.read())
                    selected_ids, split_source = select_ljspeech_ids(
                        metadata, validation_manifest, validation_size, seed
                    )
                    print(f"[INFO] Selected {len(selected_ids)} LJSpeech samples")
                    continue
                if selected_ids is None or not member.name.endswith(".wav"):
                    continue
                utterance_id = Path(member.name).stem
                if utterance_id not in selected_ids:
                    continue
                extracted_file = archive.extractfile(member)
                if extracted_file is None:
                    raise RuntimeError(f"Could not read {member.name}")
                destination = output_dir / f"{utterance_id}.wav"
                with destination.open("wb") as handle:
                    shutil.copyfileobj(extracted_file, handle)
                extracted.add(utterance_id)
                print(f"[LJSpeech] {len(extracted)}/{len(selected_ids)} {utterance_id}")

    if metadata is None or selected_ids is None or split_source is None:
        raise RuntimeError("LJSpeech metadata.csv was not found in the archive")
    missing = selected_ids - extracted
    if missing:
        raise RuntimeError(f"Missing LJSpeech audio files: {sorted(missing)[:5]}")
    rows = [
        {
            **row,
            "dataset": "ljspeech",
            "audio_path": str((output_dir / f"{row['id']}.wav").resolve()),
        }
        for row in metadata
        if row["id"] in selected_ids
    ]
    write_jsonl(output_manifest, rows)
    split_info = data_dir() / "manifests" / "ljspeech_split.json"
    split_info.write_text(json.dumps(split_source, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {len(rows)} LJSpeech rows to {output_manifest}")


def urlopen_with_retry(url: str, timeout: int = 60):
    for attempt in range(8):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            delay = min(60, 2**attempt)
            print(f"[WAIT] HTTP {error.code}; retrying in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"Repeated HTTP errors while downloading {url}")


def download_file(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen_with_retry(url) as response, partial.open("wb") as target:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        reported = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            downloaded += len(chunk)
            report_step = downloaded // (128 * 1024**2)
            if total and report_step != reported:
                reported = report_step
                print(
                    f"\r[DOWNLOAD] {destination.name}: "
                    f"{downloaded / 1024**2:.0f}/{total / 1024**2:.0f} MiB",
                    end="",
                    flush=True,
                )
    print()
    partial.replace(destination)


def prepare_vctk(force: bool) -> None:
    output_manifest = manifest_path("vctk")
    if output_manifest.exists() and not force:
        print(f"[SKIP] VCTK manifest already exists: {output_manifest}")
        return

    output_dir = data_dir() / "vctk"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_ids = set()
    print(f"[INFO] Selecting VCTK rows from {VCTK_HF_DATASET} Parquet shards")
    for shard in VCTK_SHARDS:
        shard_path = data_dir() / ".downloads" / "vctk" / f"{shard}.parquet"
        if not shard_path.exists():
            download_file(VCTK_PARQUET_URL.format(shard=shard), shard_path)
        table = pq.read_table(shard_path, columns=["speaker_id", "audio", "file", "text"])
        for source_row in table.to_pylist():
            audio_name = Path(source_row["file"]).name
            speaker = source_row["speaker_id"]
            if speaker not in VCTK_SPEAKERS or not audio_name.endswith("_mic1.flac"):
                continue
            utterance_id = audio_name.removesuffix("_mic1.flac")
            if utterance_id in seen_ids:
                continue
            destination = output_dir / speaker / audio_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if force or not destination.exists():
                destination.write_bytes(source_row["audio"]["bytes"])
            rows.append(
                {
                    "id": utterance_id,
                    "dataset": "vctk",
                    "speaker": speaker,
                    "text": source_row["text"].strip(),
                    "audio_path": str(destination.resolve()),
                }
            )
            seen_ids.add(utterance_id)
            print(f"[VCTK] {len(rows)} {utterance_id}")
        shard_path.unlink()
        print(f"[INFO] Removed temporary shard: {shard_path}")
    if not rows:
        raise RuntimeError("No selected VCTK audio files found")
    write_jsonl(output_manifest, sorted(rows, key=lambda row: row["id"]))
    print(f"[DONE] Wrote {len(rows)} VCTK rows to {output_manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ljspeech-validation-manifest", type=Path)
    parser.add_argument("--ljspeech-validation-size", type=int, default=100)
    parser.add_argument("--ljspeech-seed", type=int, default=42)
    parser.add_argument("--skip-ljspeech", action="store_true")
    parser.add_argument("--skip-vctk", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_ljspeech:
        prepare_ljspeech(
            args.ljspeech_validation_manifest,
            args.ljspeech_validation_size,
            args.ljspeech_seed,
            args.force,
        )
    if not args.skip_vctk:
        prepare_vctk(args.force)


if __name__ == "__main__":
    main()
