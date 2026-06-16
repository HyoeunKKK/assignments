"""HiFi-GAN vocoder wrapper using the pretrained model from jik876/hifi-gan."""

import json
import os
import sys
from pathlib import Path

import torch

# Add hifi-gan repo to path so we can import its Generator
HIFI_GAN_ROOT = Path("/home/elicer/project/hifi-gan")
if str(HIFI_GAN_ROOT) not in sys.path:
    sys.path.insert(0, str(HIFI_GAN_ROOT))

from models import Generator  # type: ignore  # noqa: E402


class AttrDict(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.__dict__ = self


def load_hifigan(checkpoint_dir: str, device: torch.device):
    """Load HiFi-GAN generator from a checkpoint folder containing config.json and generator_v1."""
    cdir = Path(checkpoint_dir)
    with open(cdir / "config.json") as f:
        cfg = AttrDict(json.load(f))
    gen = Generator(cfg).to(device)
    # find generator weight file
    gen_files = [p for p in cdir.iterdir() if p.name.startswith("generator")]
    assert gen_files, f"No generator file in {cdir}"
    state = torch.load(gen_files[0], map_location=device, weights_only=False)
    gen.load_state_dict(state["generator"])
    gen.eval()
    gen.remove_weight_norm()
    return gen, cfg


@torch.no_grad()
def mel_to_wav(generator, mel: torch.Tensor) -> torch.Tensor:
    """mel: (B, n_mels, T) on same device as generator. Returns audio (B, T_audio) in [-1, 1]."""
    audio = generator(mel)
    return audio.squeeze(1)
