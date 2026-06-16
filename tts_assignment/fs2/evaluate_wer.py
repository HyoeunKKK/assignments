"""Compute WER over generated audio using Whisper-large-v3-turbo."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio
import whisper
import jiwer
from tqdm import tqdm

WHISPER_SR = 16000


def load_audio_for_whisper(path: str) -> np.ndarray:
    """Load wav and return mono float32 at 16 kHz (no ffmpeg)."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != WHISPER_SR:
        wav = torchaudio.functional.resample(wav, sr, WHISPER_SR)
    return wav.squeeze(0).numpy().astype(np.float32)


def normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_whisper(name: str = "large-v3-turbo", device: torch.device | None = None):
    return whisper.load_model(name, device=str(device) if device else None)


def transcribe(model, wav_path: str) -> str:
    audio = load_audio_for_whisper(wav_path)
    result = model.transcribe(audio, language="en", fp16=torch.cuda.is_available())
    return result["text"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", required=True)
    p.add_argument("--out_jsonl", default=None)
    p.add_argument("--model", default="large-v3-turbo")
    args = p.parse_args()

    gen_dir = Path(args.gen_dir)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else gen_dir / "wer_results.jsonl"

    transcripts_path = gen_dir / "transcripts.jsonl"
    entries = []
    with open(transcripts_path) as f:
        for line in f:
            entries.append(json.loads(line))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading whisper {args.model}...")
    model = load_whisper(args.model, device)

    all_refs, all_hyps = [], []
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for e in tqdm(entries, desc="transcribe"):
            wav_path = e["audio_path"]
            hyp_raw = transcribe(model, wav_path)
            ref = normalize_text(e["transcript"])
            hyp = normalize_text(hyp_raw)
            all_refs.append(ref)
            all_hyps.append(hyp)
            row = {
                "id": e["id"],
                "ref": ref,
                "hyp": hyp,
                "hyp_raw": hyp_raw,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = jiwer.process_words(all_refs, all_hyps)
    print(f"WER: {out.wer*100:.2f}%  (S/I/D = {out.substitutions}/{out.insertions}/{out.deletions}, hits={out.hits})")

    summary = {
        "n_samples": len(all_refs),
        "wer": out.wer,
        "mer": out.mer,
        "wil": out.wil,
        "substitutions": out.substitutions,
        "insertions": out.insertions,
        "deletions": out.deletions,
        "hits": out.hits,
    }
    with open(gen_dir / "wer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary written to {gen_dir / 'wer_summary.json'}")


if __name__ == "__main__":
    main()
