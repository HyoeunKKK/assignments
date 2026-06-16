"""LJSpeech FastSpeech2 dataset."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import HOP_SIZE, SAMPLING_RATE, wav_to_mel
from .text import PAD_ID, phonemes_to_ids


def load_manifest(manifest_path: str):
    entries = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            entries[e["utterance_id"]] = e
    return entries


def load_split_ids(split_path: str):
    with open(split_path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def durations_seconds_to_frames(alignment, sr=SAMPLING_RATE, hop=HOP_SIZE):
    """Convert alignment[{start,end,...}] (seconds) -> per-phoneme frame counts.

    Uses round(start*sr/hop) and round(end*sr/hop) so sum matches total frame count.
    """
    frames = []
    for seg in alignment:
        s = int(round(seg["start"] * sr / hop))
        e = int(round(seg["end"] * sr / hop))
        frames.append(max(0, e - s))
    return frames


class LJSpeechDataset(Dataset):
    def __init__(self, manifest_path: str, split_ids_path: str, wav_root: str,
                 cache_dir: str | None = None, max_frames: int = 1000):
        self.entries_all = load_manifest(manifest_path)
        ids = load_split_ids(split_ids_path)
        self.ids = [uid for uid in ids if uid in self.entries_all]
        self.wav_root = Path(wav_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.ids)

    def _get_mel(self, uid: str) -> torch.Tensor:
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{uid}.npy"
            if cache_path.exists():
                mel = np.load(cache_path)
                return torch.from_numpy(mel)
        wav_path = self.wav_root / f"{uid}.wav"
        mel = wav_to_mel(str(wav_path))
        if self.cache_dir is not None:
            np.save(self.cache_dir / f"{uid}.npy", mel.numpy())
        return mel

    def __getitem__(self, idx):
        uid = self.ids[idx]
        e = self.entries_all[uid]
        phoneme_ids = torch.LongTensor(phonemes_to_ids(e["phoneme_sequence"]))
        durations = torch.LongTensor(durations_seconds_to_frames(e["phoneme_alignment"]))
        mel = self._get_mel(uid)  # (n_mels, T)
        # Align lengths between durations and mel frames
        total_dur = int(durations.sum().item())
        T = mel.shape[1]
        if total_dur < T:
            mel = mel[:, :total_dur]
        elif total_dur > T:
            # truncate last phoneme durations to fit
            diff = total_dur - T
            last = durations.shape[0] - 1
            while diff > 0 and last >= 0:
                take = min(diff, durations[last].item())
                durations[last] -= take
                diff -= take
                if durations[last] == 0:
                    last -= 1
        return {
            "id": uid,
            "phonemes": phoneme_ids,
            "durations": durations,
            "mel": mel,
            "text": e.get("normalized_transcript", e.get("transcript", "")),
        }


def collate_fn(batch, pad_id: int = PAD_ID):
    """Right-pad batch into tensors with masks. Returns dict."""
    B = len(batch)
    p_lens = torch.LongTensor([b["phonemes"].shape[0] for b in batch])
    m_lens = torch.LongTensor([b["mel"].shape[1] for b in batch])
    p_max = int(p_lens.max().item())
    m_max = int(m_lens.max().item())
    n_mels = batch[0]["mel"].shape[0]

    phonemes = torch.full((B, p_max), pad_id, dtype=torch.long)
    durations = torch.zeros((B, p_max), dtype=torch.long)
    mels = torch.zeros((B, n_mels, m_max), dtype=torch.float)
    ids = []
    texts = []
    for i, b in enumerate(batch):
        pl = b["phonemes"].shape[0]
        ml = b["mel"].shape[1]
        phonemes[i, :pl] = b["phonemes"]
        durations[i, :pl] = b["durations"]
        mels[i, :, :ml] = b["mel"]
        ids.append(b["id"])
        texts.append(b["text"])
    return {
        "id": ids,
        "text": texts,
        "phonemes": phonemes,
        "phoneme_lengths": p_lens,
        "durations": durations,
        "mels": mels,  # (B, n_mels, T)
        "mel_lengths": m_lens,
    }
