"""Mel-spectrogram computation matching pretrained HiFi-GAN (jik876)."""

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0

# HiFi-GAN v1 config (LJSpeech)
SAMPLING_RATE = 22050
N_FFT = 1024
NUM_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024
FMIN = 0
FMAX = 8000

_mel_basis = {}
_hann_window = {}


def load_wav(path):
    sr, data = read(path)
    return data, sr


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def mel_spectrogram(y, n_fft=N_FFT, num_mels=NUM_MELS, sampling_rate=SAMPLING_RATE,
                    hop_size=HOP_SIZE, win_size=WIN_SIZE, fmin=FMIN, fmax=FMAX, center=False):
    """Compute mel-spectrogram identically to HiFi-GAN's meldataset.py."""
    global _mel_basis, _hann_window
    device_key = str(y.device)
    fmax_key = f"{fmax}_{device_key}"
    if fmax_key not in _mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[fmax_key] = torch.from_numpy(mel).float().to(y.device)
        _hann_window[device_key] = torch.hann_window(win_size).to(y.device)

    pad = int((n_fft - hop_size) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=_hann_window[device_key], center=center,
        pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    spec = torch.matmul(_mel_basis[fmax_key], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def wav_to_mel(wav_path: str) -> torch.Tensor:
    """Load wav file and return mel-spectrogram tensor (n_mels, T)."""
    audio, sr = load_wav(wav_path)
    assert sr == SAMPLING_RATE, f"Sample rate mismatch: {sr} != {SAMPLING_RATE}"
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    audio = torch.FloatTensor(audio).unsqueeze(0)
    mel = mel_spectrogram(audio)
    return mel.squeeze(0)
