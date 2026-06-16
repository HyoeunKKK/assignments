# Assignment 3: Duration-based TTS (FastSpeech2 + HiFi-GAN)

FastSpeech2-style TTS for LJSpeech using provided IPA phoneme sequences and phoneme-level duration labels. Mel-spectrograms are generated with the duration predictor at inference, then converted to audio with a pretrained HiFi-GAN (`LJ_FT_T2_V1`).

## Project layout

```
project/
├── pyproject.toml          # uv project metadata + deps
├── fs2/
│   ├── audio.py            # mel-spec computation (HiFi-GAN compatible)
│   ├── text.py             # IPA phoneme vocabulary (46 + 4 special)
│   ├── dataset.py          # LJSpeech dataset + collate_fn
│   ├── model.py            # FastSpeech2 model
│   ├── preprocess.py       # precompute mel cache to disk
│   ├── train.py            # training loop with wandb tracking
│   ├── inference.py        # synthesize wav with predicted durations
│   ├── evaluate_wer.py     # Whisper-large-v3-turbo WER eval
│   └── vocoder.py          # HiFi-GAN wrapper
├── hifi-gan/               # cloned from jik876/hifi-gan
├── data/LJSpeech-1.1/      # raw wavs + metadata
├── cache/mels/             # precomputed mel-spectrograms (npy)
├── checkpoints/
│   └── hifi-gan-pretrained/LJ_FT_T2_V1/
└── runs/fs2/               # training output (checkpoints + wandb)
```

## Setup

Uses [uv](https://github.com/astral-sh/uv).

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install deps
cd project
uv sync
```

PyTorch is pinned to `2.5.1+cu121` for compatibility with CUDA 12.2 drivers.

### Data preparation

```bash
# 1) Download LJSpeech and extract to project/data/LJSpeech-1.1
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -O data/LJSpeech-1.1.tar.bz2
tar -xjf data/LJSpeech-1.1.tar.bz2 -C data/

# 2) Use the provided manifest (placed in SLP_assignment03_LJSpeech_manifest/manifests/)
#    Contains compressed IPA phoneme sequence + phoneme-level alignments.

# 3) Pretrained HiFi-GAN (jik876/hifi-gan, LJ_FT_T2_V1)
gdown --folder https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y -O checkpoints/hifi-gan-pretrained

# 4) Precompute mel-spectrograms (~2.3 GB)
uv run python -m fs2.preprocess --workers 8
```

### Training

```bash
export WANDB_API_KEY=<your_key>

uv run python -m fs2.train \
    --batch_size 32 \
    --max_steps 50000 \
    --val_interval_steps 1000 \
    --ckpt_interval_steps 2000 \
    --log_interval 50 \
    --num_workers 4 \
    --wandb_project slp-fs2-ljspeech \
    --wandb_run my-run
```

Training tracks `loss`, `mel_before/after`, `duration`, `lr`, and `grad_norm` on wandb.

### Inference (uses predicted durations)

```bash
uv run python -m fs2.inference \
    --ckpt runs/fs2/latest.pt \
    --vocoder checkpoints/hifi-gan-pretrained/LJ_FT_T2_V1 \
    --split test \
    --out_dir runs/fs2/gen_test
```

### Five specific samples

```bash
uv run python -m fs2.inference \
    --ckpt runs/fs2/latest.pt \
    --ids LJ012-0091 LJ014-0066 LJ003-0011 LJ002-0272 LJ015-0205 \
    --out_dir runs/fs2/gen_five
```

### WER (Whisper-large-v3-turbo)

```bash
uv run python -m fs2.evaluate_wer --gen_dir runs/fs2/gen_test
```

This writes `wer_results.jsonl` (per-utterance) and `wer_summary.json`.

## Model

| Component         | Setting                                  |
|-------------------|------------------------------------------|
| Phoneme embedding | vocab = 50 (46 IPA + pad/bos/eos/unk)    |
| Encoder           | 4 × FFT block (2-head MHA + 9-kernel ConvFFN), d=256, ff=1024 |
| Variance adaptor  | Duration predictor (2 × 3-kernel Conv1d, d=256, dropout=0.5) |
| Length regulator  | repeat hidden states by duration         |
| Decoder           | 4 × FFT block (same config as encoder)   |
| Mel head          | Linear 256 → 80                          |
| Postnet           | 5 × 1D-Conv (Tanh+BN+dropout)            |

Total: **41.49M trainable parameters**.

Mel-spec config (matches HiFi-GAN v1 LJ):
- sample_rate=22050, n_fft=1024, hop_size=256, win_size=1024
- 80 mel bins, fmin=0, fmax=8000, log-magnitude (dynamic range compression)

## Losses

- `mel_before` (L1 over mel before postnet, masked over real frames)
- `mel_after` (L1 over mel after postnet, masked over real frames)
- `duration` (MSE between predicted log-duration and `log(d_gt + 1)`, masked over real phonemes)
- total = sum of the three

Encoder hidden states are **detached** before being fed to the duration predictor, so duration loss does not update the encoder.

## Optimization

- AdamW (β₁=0.9, β₂=0.98, eps=1e-9, wd=1e-6)
- Noam schedule (d_model=256, warmup=4000 steps)
- Gradient clip = 1.0
