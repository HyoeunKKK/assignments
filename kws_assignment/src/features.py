import torch
import torch.nn as nn
import torchaudio


class LogMelExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 640,
        win_length: int = 640,
        hop_length: int = 320,
        n_mels: int = 80,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [T] or [1, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        feat = self.mel(waveform)      # [1, 80, T]
        feat = self.to_db(feat)        # [1, 80, T]
        feat = feat.squeeze(0)         # [80, T]

        feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        return feat