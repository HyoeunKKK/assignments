"""Inference: generate audio from test set phonemes using predicted durations + HiFi-GAN."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from .audio import SAMPLING_RATE
from .dataset import load_manifest, load_split_ids
from .model import FastSpeech2
from .text import PAD_ID, VOCAB_SIZE, phonemes_to_ids
from .vocoder import load_hifigan, mel_to_wav


def load_fs2(ckpt_path: str, device: torch.device, args_override: dict | None = None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    if args_override:
        train_args.update(args_override)
    model = FastSpeech2(
        vocab_size=VOCAB_SIZE,
        d_model=train_args.get("d_model", 256),
        n_heads=train_args.get("n_heads", 2),
        d_ff=train_args.get("d_ff", 1024),
        enc_layers=train_args.get("enc_layers", 4),
        dec_layers=train_args.get("dec_layers", 4),
        pad_id=PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, train_args


@torch.no_grad()
def synthesize_one(model, vocoder, phoneme_ids, device, duration_scale: float = 1.0):
    phonemes = torch.LongTensor(phoneme_ids).unsqueeze(0).to(device)
    lengths = torch.LongTensor([len(phoneme_ids)]).to(device)
    out = model(phonemes, lengths)
    if duration_scale != 1.0:
        # rerun with scaled durations
        dur = out["durations_used"].float() * duration_scale
        dur = torch.clamp(torch.round(dur), min=0).long()
        # re-expand: easier to just call internal pieces; we recompute by passing durations as gt
        mel_lengths_fake = dur.sum(dim=1)
        out = model(phonemes, lengths, durations=dur, mel_lengths=mel_lengths_fake)
    mel = out["mel_after"]  # (1, T, n_mels)
    mel = mel.transpose(1, 2)  # (1, n_mels, T)
    audio = mel_to_wav(vocoder, mel)  # (1, T_audio)
    return audio.squeeze(0).cpu(), mel.cpu()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocoder", default="/home/elicer/project/checkpoints/hifi-gan-pretrained/LJ_FT_T2_V1")
    p.add_argument("--manifest", default="/home/elicer/SLP_assignment03_LJSpeech_manifest/manifests/ljspeech_ipa_compressed.jsonl")
    p.add_argument("--splits_dir", default="/home/elicer/SLP_assignment03_LJSpeech_manifest/manifests/splits")
    p.add_argument("--split", default="test")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--ids", nargs="*", default=None,
                   help="If provided, only synthesize these utterance IDs")
    p.add_argument("--duration_scale", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading vocoder...")
    vocoder, _ = load_hifigan(args.vocoder, device)
    print("loading fs2...")
    model, train_args = load_fs2(args.ckpt, device)
    print("ready.")

    entries = load_manifest(args.manifest)
    if args.ids:
        ids = args.ids
    else:
        split_file = os.path.join(args.splits_dir, f"{args.split}_ids.txt")
        ids = load_split_ids(split_file)
        if args.limit > 0:
            ids = ids[:args.limit]

    transcripts_path = out_dir / "transcripts.jsonl"
    with open(transcripts_path, "w", encoding="utf-8") as tf:
        for uid in tqdm(ids, desc="synthesize"):
            if uid not in entries:
                continue
            e = entries[uid]
            pids = phonemes_to_ids(e["phoneme_sequence"])
            audio, mel = synthesize_one(model, vocoder, pids, device, args.duration_scale)
            torchaudio.save(str(out_dir / f"{uid}.wav"), audio.unsqueeze(0), SAMPLING_RATE)
            tf.write(json.dumps({
                "id": uid,
                "transcript": e.get("normalized_transcript", e.get("transcript", "")),
                "n_phonemes": len(pids),
                "audio_path": str(out_dir / f"{uid}.wav"),
            }, ensure_ascii=False) + "\n")

    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
