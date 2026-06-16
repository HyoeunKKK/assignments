"""Stage 1: ASR. Transcribe all questions + documents once and cache to disk.

Usage:
    python transcribe.py --split dev
    python transcribe.py --split release

Writes cache/{split}_questions.json and cache/{split}_documents.json
mapping id -> transcript (e.g. {"q000": "...", ...}).
Re-running skips ids already present in the cache.
"""
import argparse
import glob
import json
import os

import config
from faster_whisper import WhisperModel


def wav_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def transcribe_files(model, files, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    todo = [f for f in files if wav_id(f) not in cache]
    print(f"  {len(cache)} cached, {len(todo)} to transcribe -> {cache_path}")

    for i, path in enumerate(todo, 1):
        segments, _ = model.transcribe(
            path,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        cache[wav_id(path)] = text
        if i % 25 == 0 or i == len(todo):
            print(f"    [{i}/{len(todo)}] {wav_id(path)}: {text[:60]!r}")
            with open(cache_path, "w") as f:
                json.dump(cache, f, ensure_ascii=False, indent=0)

    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(config.SPLITS), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="int8_float16")
    args = ap.parse_args()

    print(f"Loading faster-whisper {config.ASR_MODEL} "
          f"({args.compute_type}) ...")
    model = WhisperModel(
        config.ASR_MODEL,
        device=args.device,
        compute_type=args.compute_type,
        download_root=os.path.join(config.HF_HOME, "faster_whisper"),
    )

    for kind, tag in (("questions", "questions"), ("documents", "documents")):
        files = sorted(glob.glob(os.path.join(config.split_dir(args.split, kind), "*.wav")))
        cache_path = os.path.join(config.CACHE_DIR, f"{args.split}_{tag}.json")
        print(f"[{args.split}/{kind}] {len(files)} files")
        transcribe_files(model, files, cache_path)

    print("done.")


if __name__ == "__main__":
    main()
