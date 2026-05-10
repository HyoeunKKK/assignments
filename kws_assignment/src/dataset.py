import csv

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from torch.utils.data import Dataset

from src.augment import pad_or_trim, waveform_augment, spec_augment
from src.features import LogMelExtractor

LABELS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "silence", "unknown"
]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


class GSCDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        clip_samples: int = 16000,
        n_fft: int = 640,
        win_length: int = 640,
        hop_length: int = 320,
        n_mels: int = 80,
        train: bool = False,
    ):
        self.records = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.train = train

        self.extractor = LogMelExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __len__(self):
        return len(self.records)

    def _load_audio(self, path: str) -> torch.Tensor:
        if path == "__silence__":
            return torch.zeros(self.clip_samples, dtype=torch.float32)

        sr, wav = wavfile.read(path)
        wav = self._to_float32(wav)
        wav = torch.from_numpy(wav)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, self.sample_rate
            ).squeeze(0)

        wav = pad_or_trim(wav, self.clip_samples)
        return wav

    @staticmethod
    def _to_float32(wav: np.ndarray) -> np.ndarray:
        if np.issubdtype(wav.dtype, np.floating):
            return wav.astype(np.float32, copy=False)

        if wav.dtype == np.int16:
            return wav.astype(np.float32) / 32768.0
        if wav.dtype == np.int32:
            return wav.astype(np.float32) / 2147483648.0
        if wav.dtype == np.uint8:
            return (wav.astype(np.float32) - 128.0) / 128.0

        return wav.astype(np.float32)

    def __getitem__(self, idx):
        row = self.records[idx]
        path = row["path"]
        label = row["label"]

        waveform = self._load_audio(path)
        is_silence = (label == "silence")

        if self.train and not is_silence:
            waveform = waveform_augment(
                waveform,
                sample_rate=self.sample_rate,
                target_len=self.clip_samples,
            )

        feat = self.extractor(waveform)  # [80, T]

        if self.train and not is_silence:
            feat = spec_augment(feat)

        feat = feat.unsqueeze(0)  # [1, 80, T]
        target = LABEL_TO_IDX[label]
        return feat, target

