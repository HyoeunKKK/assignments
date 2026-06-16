"""End-to-end evaluation: synthesize 500 test samples, compute WER, and synthesize 5 specific samples."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from .audio import SAMPLING_RATE
from .dataset import load_manifest, load_split_ids
from .inference import load_fs2, synthesize_one
from .text import phonemes_to_ids
from .vocoder import load_hifigan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/home/elicer/project/runs/fs2/latest.pt")
    p.add_argument("--vocoder", default="/home/elicer/project/checkpoints/hifi-gan-pretrained/LJ_FT_T2_V1")
    p.add_argument("--manifest", default="/home/elicer/SLP_assignment03_LJSpeech_manifest/manifests/ljspeech_ipa_compressed.jsonl")
    p.add_argument("--splits_dir", default="/home/elicer/SLP_assignment03_LJSpeech_manifest/manifests/splits")
    p.add_argument("--out_root", default="/home/elicer/project/runs/fs2/eval")
    p.add_argument("--five_ids", nargs="*",
                   default=["LJ012-0091", "LJ014-0066", "LJ003-0011", "LJ002-0272", "LJ015-0205"])
    p.add_argument("--skip_synth", action="store_true")
    p.add_argument("--skip_wer", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    test_out = out_root / "test_500"
    five_out = out_root / "five_samples"
    test_out.mkdir(parents=True, exist_ok=True)
    five_out.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(args.manifest)
    test_ids = load_split_ids(os.path.join(args.splits_dir, "test_ids.txt"))

    if not args.skip_synth:
        print("loading vocoder...")
        vocoder, _ = load_hifigan(args.vocoder, device)
        print("loading fs2...")
        model, _ = load_fs2(args.ckpt, device)

        # 500 test
        print(f"synthesizing {len(test_ids)} test samples -> {test_out}")
        t0 = time.time()
        with open(test_out / "transcripts.jsonl", "w", encoding="utf-8") as tf:
            for uid in tqdm(test_ids):
                if uid not in entries:
                    continue
                e = entries[uid]
                pids = phonemes_to_ids(e["phoneme_sequence"])
                audio, _ = synthesize_one(model, vocoder, pids, device)
                torchaudio.save(str(test_out / f"{uid}.wav"), audio.unsqueeze(0), SAMPLING_RATE)
                tf.write(json.dumps({
                    "id": uid,
                    "transcript": e.get("normalized_transcript", e.get("transcript", "")),
                    "audio_path": str(test_out / f"{uid}.wav"),
                }, ensure_ascii=False) + "\n")
        print(f"500 test done in {(time.time()-t0)/60:.1f}min")

        # 5 specific samples
        print(f"synthesizing 5 specific samples -> {five_out}")
        with open(five_out / "transcripts.jsonl", "w", encoding="utf-8") as tf:
            for uid in args.five_ids:
                if uid not in entries:
                    print(f"!! {uid} not in manifest")
                    continue
                e = entries[uid]
                pids = phonemes_to_ids(e["phoneme_sequence"])
                audio, _ = synthesize_one(model, vocoder, pids, device)
                torchaudio.save(str(five_out / f"{uid}.wav"), audio.unsqueeze(0), SAMPLING_RATE)
                tf.write(json.dumps({
                    "id": uid,
                    "transcript": e.get("normalized_transcript", e.get("transcript", "")),
                    "audio_path": str(five_out / f"{uid}.wav"),
                }, ensure_ascii=False) + "\n")
        print(f"5 samples synthesized.")

    if not args.skip_wer:
        # Reuse evaluate_wer logic by importing
        from .evaluate_wer import normalize_text, load_whisper, transcribe
        import jiwer

        entries_path = test_out / "transcripts.jsonl"
        items = [json.loads(l) for l in open(entries_path)]
        print(f"loading whisper large-v3-turbo on {device}...")
        whisper_model = load_whisper("large-v3-turbo", device)
        refs, hyps = [], []
        wer_jsonl = test_out / "wer_results.jsonl"
        with open(wer_jsonl, "w", encoding="utf-8") as fout:
            for e in tqdm(items, desc="transcribe"):
                hyp_raw = transcribe(whisper_model, e["audio_path"])
                r = normalize_text(e["transcript"])
                h = normalize_text(hyp_raw)
                refs.append(r); hyps.append(h)
                fout.write(json.dumps({"id": e["id"], "ref": r, "hyp": h, "hyp_raw": hyp_raw},
                                       ensure_ascii=False) + "\n")
        m = jiwer.process_words(refs, hyps)
        summary = {
            "n_samples": len(refs),
            "wer": m.wer,
            "mer": m.mer,
            "wil": m.wil,
            "substitutions": m.substitutions,
            "insertions": m.insertions,
            "deletions": m.deletions,
            "hits": m.hits,
        }
        with open(test_out / "wer_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"WER = {summary['wer']*100:.2f}% (n={summary['n_samples']})")
        print(json.dumps(summary, indent=2))

    print(f"all outputs in {out_root}")


if __name__ == "__main__":
    main()
