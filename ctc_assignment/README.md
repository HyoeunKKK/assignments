# Assignment 02 — CTC-based ASR

Fine-tunes a CTC head (BiLSTM + Linear) on top of pre-extracted W2V2-BERT-2.0 hidden states.

## Setup

```bash
uv sync
# or: pip install torch numpy tqdm wandb
```

## Training

```bash
python train.py \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --fp16 \
    --cache_ram
```

Key arguments:

| Flag | Default | Description |
|---|---|---|
| `--batch_size` | 64 | Batch size |
| `--epochs` | 100 | Number of epochs |
| `--lr` | 1e-3 | Peak learning rate (linear warmup + cosine decay) |
| `--warmup_ratio` | 0.05 | Fraction of total steps for linear warmup |
| `--dropout` | 0.1 | Dropout rate |
| `--fp16` | True | Mixed precision (AMP) |
| `--cache_ram` | False | Load all features into RAM (recommended for S3/slow disk) |
| `--wandb_project` | `ctc-asr` | WandB project name |
| `--wandb_run` | `w2v2bert-ctc` | WandB run name |
| `--ckpt_dir` | `checkpoints/` | Checkpoint directory |

Training metrics (loss, WER, LR, grad norm) are logged to WandB.

## Evaluation

```bash
python evaluate.py --ckpt checkpoints/best_model.pt
```

Reports WER on `test-clean` and `test-other` with sample predictions.

**Final results:**

| Split | WER |
|---|---|
| test-clean | 17.06% |
| test-other | 35.84% |

## Data

Pre-extracted hidden states:
- Features: `/mnt/elice/datahub/speech-dataset/cached_features/{split}/*.npz`
- Transcripts: `/mnt/elice/datahub/speech-dataset/librispeech/LibriSpeech/{split}/`

Each `.npz` file contains:
- `hidden_states`: `(8, T, 1024)` float16 — 8 encoder layers of W2V2-BERT-2.0
- `seq_len`: int64

## Model

```
hidden_states (B, 8, T, 1024)
  → learnable weighted sum (Softmax normalised, 8 scalar weights)
  → (B, T, 1024)
  → LayerNorm + Dropout
  → 2-layer BiLSTM (hidden=512, bidirectional → 1024 out)
  → Dropout
  → Linear(1024, 29)
  → log_softmax → CTCLoss
```

Vocabulary (29 tokens): `<blank>`, a–z, `'`, ` `

## Project Structure

```
assignment02/
├── utils.py       # Vocabulary, encode/decode, WER
├── dataset.py     # Dataset and DataLoader
├── model.py       # CTCModel (weighted sum + BiLSTM + Linear)
├── train.py       # Training loop with WandB logging
├── evaluate.py    # Evaluation and sample predictions
└── pyproject.toml
```
