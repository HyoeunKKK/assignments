# NAC Assignment

This project evaluates RVQ-level degradation using the pretrained DAC 24 kHz
checkpoint. It produces one metric table for LJSpeech validation audio and one
for VCTK speakers `p232` and `p257`.

## Fixed experiment settings

- Codec: official DAC 24 kHz pretrained checkpoint
- RVQ settings: original wav, full RVQ, 5, 3, 2, and 1 RVQ levels
- Codec input: mono audio resampled to 24 kHz
- LogMel difference: L1 mean over an 80-bin log Mel spectrogram, FFT/window
  size 1024, hop length 256, and frequency range 0-12,000 Hz
- WER: Whisper `large`, English transcription, beam size `1` for every setting
  and decode batch size `2`
- Naturalness: official open-source UTMOSv2 pretrained model, inference batch
  size `8`
- GPU: physical GPU `0`, exposed as `cuda:0`

## Environment

The working environment is intentionally isolated from other projects:

```bash
export A04_WORKSPACE=/path/to/nac_assignment_workspace
export CUDA_VISIBLE_DEVICES=0
export HOME="$A04_WORKSPACE/cache/home"
export XDG_CACHE_HOME="$A04_WORKSPACE/cache/xdg"
export HF_HOME="$A04_WORKSPACE/cache/huggingface"
export TORCH_HOME="$A04_WORKSPACE/cache/torch"
export PIP_CACHE_DIR="$A04_WORKSPACE/cache/pip"
export UTMOSV2_CHACHE="$A04_WORKSPACE/models/utmosv2_cache"
```

To create an environment:

```bash
python -m venv "$A04_WORKSPACE/env"
"$A04_WORKSPACE/env/bin/pip" install -r requirements.txt
```

## Run

The LJSpeech split from the previous assignment can be passed in any text
format containing LJSpeech IDs such as `LJ001-0001`:

```bash
"$A04_WORKSPACE/env/bin/python" prepare_data.py \
  --ljspeech-validation-manifest /path/to/previous_assignment_validation_manifest.txt
```

When the previous manifest is unavailable, `prepare_data.py` creates a
deterministic fallback validation split of 100 clips with seed `42`. The exact
split rule is written to `data/manifests/ljspeech_split.json`.

Run the remaining stages:

```bash
"$A04_WORKSPACE/env/bin/python" run_codec.py --device cuda:0
"$A04_WORKSPACE/env/bin/python" evaluate.py \
  --device cuda:0 --whisper-model large --whisper-batch-size 2
```

`run_all.sh` executes data preparation, codec reconstruction, WER, UTMOS, and
metric aggregation sequentially. Metric JSONL files are append-only where
practical, so WER and UTMOS evaluation can resume after an interruption.

VCTK is selected through the Hugging Face Dataset Viewer API from
`sanchit-gandhi/vctk`, a Parquet conversion of the standard
`wav48_silence_trimmed` corpus. The script downloads only `p232` and `p257`
`mic1` audio from the relevant temporary shards instead of retaining the full
10.94 GiB official archive.

## Outputs

- `outputs/audio/`: reconstructed waveforms
- `outputs/metrics/codec/`: per-file LogMel differences
- `outputs/metrics/wer/`: Whisper transcriptions and WER inputs
- `outputs/metrics/utmos/`: per-file UTMOSv2 predictions
- `outputs/summary/`: CSV, JSON, and Markdown metric tables

Downloaded models, data, reconstructed audio, reports, and the environment are
not included in this repository.
