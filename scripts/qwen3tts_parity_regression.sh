#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-weights/qwen3-tts-0.6b-base}"
if [[ -n "${QWEN3_TTS_PARITY_AUDIO:-}" ]]; then
  AUDIO_PATH="${QWEN3_TTS_PARITY_AUDIO}"
elif [[ -f output/mlk_10s.wav ]]; then
  AUDIO_PATH="output/mlk_10s.wav"
else
  AUDIO_PATH="MLKDream.wav"
fi
QWEN_REPO="${QWEN3_TTS_REPO:-../Qwen3-TTS}"
OUT_DIR="${QWEN3_TTS_PARITY_OUT_DIR:-output/parity_regression}"
LEAN_CODES="$OUT_DIR/lean.codes"
PY_CODES="$OUT_DIR/python.codes"
TYR_DEVICE="${TYR_DEVICE:-mps}"
DEVICE_MAP="${QWEN3_TTS_DEVICE_MAP:-mps}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[qwen3tts-parity] skip: model dir not found: $MODEL_DIR"
  exit 0
fi
if [[ ! -f "$AUDIO_PATH" ]]; then
  echo "[qwen3tts-parity] skip: audio file not found: $AUDIO_PATH"
  exit 0
fi
if [[ ! -d "$QWEN_REPO" ]]; then
  echo "[qwen3tts-parity] skip: qwen repo not found: $QWEN_REPO"
  exit 0
fi
if [[ ! -f scripts/qwen3tts_encode_audio.py ]]; then
  echo "[qwen3tts-parity] skip: missing scripts/qwen3tts_encode_audio.py"
  exit 0
fi

mkdir -p "$OUT_DIR"

echo "[qwen3tts-parity] building Lean executable"
uv run lake build Qwen3TTSEndToEnd >/dev/null

echo "[qwen3tts-parity] Lean encode"
TYR_DEVICE="$TYR_DEVICE" uv run lake env ./.lake/build/bin/Qwen3TTSEndToEnd \
  --model-dir "$MODEL_DIR" \
  --encode-audio-path "$AUDIO_PATH" \
  --encode-out-codes-path "$LEAN_CODES" \
  --encode-only >/dev/null

echo "[qwen3tts-parity] Python reference encode"
uv run python scripts/qwen3tts_encode_audio.py \
  --speech-tokenizer-dir "$MODEL_DIR/speech_tokenizer" \
  --audio "$AUDIO_PATH" \
  --output-codes "$PY_CODES" \
  --device-map "$DEVICE_MAP" \
  --qwen3-tts-repo "$QWEN_REPO" >/dev/null

PREFIX_ROWS="${QWEN3_TTS_PARITY_PREFIX_ROWS:-125}"
PREFIX_TOKEN_MIN="${QWEN3_TTS_PARITY_PREFIX_TOKEN_MIN:-0.99}"
PREFIX_ROW_MIN="${QWEN3_TTS_PARITY_PREFIX_ROW_MIN:-0.99}"
FULL_TOKEN_MIN="${QWEN3_TTS_PARITY_FULL_TOKEN_MIN:-0.10}"
NONZERO_MIN="${QWEN3_TTS_PARITY_NONZERO_MIN:-0.90}"

uv run python - "$LEAN_CODES" "$PY_CODES" \
  "$PREFIX_ROWS" "$PREFIX_TOKEN_MIN" "$PREFIX_ROW_MIN" "$FULL_TOKEN_MIN" "$NONZERO_MIN" <<'PY'
import sys
from pathlib import Path
import numpy as np

lean_path, py_path, prefix_rows_s, p_tok_s, p_row_s, full_tok_s, nonzero_s = sys.argv[1:]
prefix_rows = int(prefix_rows_s)
prefix_tok_min = float(p_tok_s)
prefix_row_min = float(p_row_s)
full_tok_min = float(full_tok_s)
nonzero_min = float(nonzero_s)

def read_codes(path: str) -> np.ndarray:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [[int(x) for x in ln.split()] for ln in lines]
    return np.asarray(rows, dtype=np.int64)

lean = read_codes(lean_path)
py = read_codes(py_path)
if lean.shape != py.shape:
    print(f"[qwen3tts-parity] shape mismatch lean={lean.shape} py={py.shape}")
    sys.exit(1)

match = (lean == py)
full_tok = float(match.mean())
full_row = float(match.all(axis=1).mean())
nonzero_ratio = float((lean != 0).mean())

pref_n = min(prefix_rows, lean.shape[0])
pref = match[:pref_n]
pref_tok = float(pref.mean())
pref_row = float(pref.all(axis=1).mean())

print(f"[qwen3tts-parity] shape={lean.shape}")
print(f"[qwen3tts-parity] prefix_rows={pref_n} prefix_token_match={pref_tok:.6f} prefix_row_exact={pref_row:.6f}")
print(f"[qwen3tts-parity] full_token_match={full_tok:.6f} full_row_exact={full_row:.6f} nonzero_ratio={nonzero_ratio:.6f}")

ok = True
if pref_tok < prefix_tok_min:
    print(f"[qwen3tts-parity] FAIL: prefix token match {pref_tok:.6f} < {prefix_tok_min:.6f}")
    ok = False
if pref_row < prefix_row_min:
    print(f"[qwen3tts-parity] FAIL: prefix row exact {pref_row:.6f} < {prefix_row_min:.6f}")
    ok = False
if full_tok < full_tok_min:
    print(f"[qwen3tts-parity] FAIL: full token match {full_tok:.6f} < {full_tok_min:.6f}")
    ok = False
if nonzero_ratio < nonzero_min:
    print(f"[qwen3tts-parity] FAIL: nonzero ratio {nonzero_ratio:.6f} < {nonzero_min:.6f}")
    ok = False

if not ok:
    sys.exit(1)
PY

echo "[qwen3tts-parity] PASS"
