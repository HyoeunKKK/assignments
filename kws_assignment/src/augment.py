import random
import torch
import torchaudio


def pad_or_trim(waveform: torch.Tensor, target_len: int = 16000) -> torch.Tensor:
    if waveform.numel() < target_len:
        pad_len = target_len - waveform.numel()
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    elif waveform.numel() > target_len:
        waveform = waveform[:target_len]
    return waveform

def random_speed_perturb(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    target_len: int = 16000,
    speeds: tuple[float, ...] = (0.9, 1.0, 1.1),
    p: float = 0.5,
) -> torch.Tensor:
    if random.random() > p:
        return waveform

    speed = random.choice(speeds)
    if speed == 1.0:
        return waveform

    source_sr = int(sample_rate * speed)

    # speed > 1.0 이면 더 빠르게, < 1.0 이면 느리게
    out = torchaudio.functional.resample(
        waveform.unsqueeze(0),
        orig_freq=source_sr,
        new_freq=sample_rate,
    ).squeeze(0)

    out = pad_or_trim(out, target_len)
    return out

def random_time_shift(waveform: torch.Tensor, shift_limit: int = 1600) -> torch.Tensor:
    shift = random.randint(-shift_limit, shift_limit)
    if shift == 0:
        return waveform

    out = torch.zeros_like(waveform)
    if shift > 0:
        out[shift:] = waveform[:-shift]
    else:
        out[:shift] = waveform[-shift:]
    return out


def random_gain_noise(waveform: torch.Tensor, noise_std: float = 0.003) -> torch.Tensor:
    gain = random.uniform(0.85, 1.15)
    noise = torch.randn_like(waveform) * noise_std
    out = waveform * gain + noise
    return out.clamp(-1.0, 1.0)


def waveform_augment(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    target_len: int = 16000,
) -> torch.Tensor:
    waveform = random_speed_perturb(
        waveform,
        sample_rate=sample_rate,
        target_len=target_len,
        speeds=(0.9, 1.0, 1.1),
        p=0.5,
    )
    waveform = random_time_shift(waveform, shift_limit=1600)  # ±100ms
    waveform = random_gain_noise(waveform, noise_std=0.003)
    waveform = pad_or_trim(waveform, target_len)
    return waveform


def spec_augment(
    feat: torch.Tensor,
    max_freq_mask: int = 10,
    max_time_mask: int = 8,
) -> torch.Tensor:
    # feat: [80, T]
    out = feat.clone()
    freq_bins, time_steps = out.shape

    f = random.randint(0, max_freq_mask)
    if f > 0 and f < freq_bins:
        f0 = random.randint(0, freq_bins - f)
        out[f0:f0 + f, :] = 0.0

    t = random.randint(0, max_time_mask)
    if t > 0 and t < time_steps:
        t0 = random.randint(0, time_steps - t)
        out[:, t0:t0 + t] = 0.0

    return out
