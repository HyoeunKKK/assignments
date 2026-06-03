#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${A04_WORKSPACE:-$SCRIPT_DIR}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$WORKSPACE/env/bin/python" ]]; then
    PYTHON="$WORKSPACE/env/bin/python"
  else
    PYTHON=python
  fi
fi

export A04_WORKSPACE="$WORKSPACE"
export CUDA_VISIBLE_DEVICES=0
export HOME="$WORKSPACE/cache/home"
export XDG_CACHE_HOME="$WORKSPACE/cache/xdg"
export HF_HOME="$WORKSPACE/cache/huggingface"
export TORCH_HOME="$WORKSPACE/cache/torch"
export PIP_CACHE_DIR="$WORKSPACE/cache/pip"
export UTMOSV2_CHACHE="$WORKSPACE/models/utmosv2_cache"

"$PYTHON" "$SCRIPT_DIR/prepare_data.py" "$@"
"$PYTHON" "$SCRIPT_DIR/run_codec.py" --device cuda:0
"$PYTHON" "$SCRIPT_DIR/evaluate.py" --device cuda:0 --whisper-model large --whisper-batch-size 2
