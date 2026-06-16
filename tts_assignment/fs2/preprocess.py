"""Precompute and cache mel-spectrograms for all LJSpeech wav files."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .audio import wav_to_mel


def process_one(args):
    uid, wav_path, cache_dir = args
    out_path = Path(cache_dir) / f"{uid}.npy"
    if out_path.exists():
        return uid, True
    mel = wav_to_mel(wav_path)
    np.save(out_path, mel.numpy())
    return uid, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/home/elicer/SLP_assignment03_LJSpeech_manifest/manifests/ljspeech_ipa_compressed.jsonl")
    p.add_argument("--wav_root", default="/home/elicer/project/data/LJSpeech-1.1/wavs")
    p.add_argument("--cache_dir", default="/home/elicer/project/cache/mels")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    tasks = []
    with open(args.manifest) as f:
        for line in f:
            e = json.loads(line)
            uid = e["utterance_id"]
            wp = os.path.join(args.wav_root, f"{uid}.wav")
            tasks.append((uid, wp, args.cache_dir))

    print(f"processing {len(tasks)} files with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()
    print("done.")


if __name__ == "__main__":
    main()
