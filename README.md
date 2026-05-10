# Project 1: Streaming ASR Subtitle System

Spoken Language Processing — Sungkyunkwan University, 2026

This project implements a streaming ASR subtitle system for two-speaker conversation videos. The system displays speaker-attributed subtitles in real time and saves committed subtitles as JSON annotation files.

## Requirements

- Python 3.10+
- PyTorch with CUDA
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) or `openai-whisper`
- `silero-vad` (loaded via `torch.hub`)
- `pyannote.audio` (for speaker embeddings)
- `ffmpeg` (for video encoding)

Install dependencies (recommended with [uv](https://github.com/astral-sh/uv)):

```bash
uv pip install torch openai-whisper pyannote.audio silero
```

Set your HuggingFace token for pyannote:

```bash
export HF_TOKEN=your_hf_token
```

## How to Run

```bash
python streaming_asr/streaming_asr.py \
  --clip all \
  --data_dir /path/to/SLP_project01_data \
  --output_dir ./output_submit \
  --model medium.en
```

| Argument | Default | Description |
|---|---|---|
| `--clip` | `all` | Clip IDs to process (`all` or comma-separated, e.g. `00,01`) |
| `--data_dir` | `SLP_project01_data/` | Directory containing raw videos and speaker embeddings |
| `--output_dir` | `output_submit/` | Output directory |
| `--model` | `medium.en` | Whisper model (`small.en` or `medium.en`) |

**Example — single clip:**
```bash
python streaming_asr/streaming_asr.py --clip 00 --data_dir SLP_project01_data --output_dir output_submit --model medium.en
```

## Outputs

For each clip `XX`, the script writes:

- `clipXX_subtitled.mp4` — video with live-style ASS subtitles burned in
- `clipXX_annotation.json` — committed subtitle annotation (`speaker`, `start`, `end`, `commit_time`, `text`)
- `clipXX_subtitled.srt` — SRT subtitle file
- `clipXX_subtitled.ass` — ASS subtitle file (used for rendering)

A combined `project1_annotation.json` covering clips 00–04 can be generated with:
```bash
python create_report_docx.py   # also writes project1_report.docx
```

## Streaming Constraints (per project spec)

| Component | Max frequency |
|---|---|
| Whisper `small.en` | 2 Hz |
| Whisper `medium.en` | 1 Hz |
| Silero VAD | 20 Hz |
| Pyannote speaker embedding | 5 Hz |

The system only accesses audio available up to the current playback time — no full-video offline transcription.

## System Design

```
Audio chunks (0.5 s)
  → Silero VAD       (20 Hz) — detect speech regions
  → Pyannote embed.  ( 5 Hz) — assign Speaker A / B
  → Whisper ASR      ( 1 Hz) — decode buffered speech
  → Local Agreement  (N=2)   — commit stable prefix
  → Attn. truncation         — drop right-boundary tokens
  → ASS renderer             — burn subtitles into video
```

**Key techniques:**
- **Step 1 — Chunk-wise streaming**: causal 0.5 s audio buffer, VAD endpointing, 1.2 s pre-roll for onset
- **Step 2 — Local Agreement**: words committed only when stable across ≥ 2 consecutive ASR hypotheses
- **Step 3 — Attention-based truncation**: Whisper decoder cross-attention used to detect and drop tokens attending near the right audio boundary (Simul-Whisper)
- Speaker turn pre-roll (0.4 s) and overlap-strip for clean turn transitions
- Backchannel / one-word question overlay handling
- Monotonic committed text display (white = committed, blue = partial)
