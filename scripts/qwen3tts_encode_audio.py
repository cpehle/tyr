#!/usr/bin/env python3
"""
Encode audio into Qwen3-TTS speech-tokenizer codec IDs.

Output `codes` format:
  One frame per line, whitespace-separated codebook IDs.
  For 12Hz tokenizer this is typically 16 columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_qwen_tokenizer(repo_path: Path):
    try:
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

        return Qwen3TTSTokenizer
    except Exception:
        sys.path.insert(0, str(repo_path))
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

        return Qwen3TTSTokenizer


def _save_codes(path: Path, codes) -> None:
    if codes.dim() == 1:
        codes = codes.unsqueeze(-1)
    if codes.dim() != 2:
        raise ValueError(f"Expected 1D or 2D codec tensor, got shape={tuple(codes.shape)}")

    rows = codes.detach().cpu().tolist()
    lines = [" ".join(str(int(v)) for v in row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech-tokenizer-dir", required=True)
    parser.add_argument("--audio", required=True, help="Input audio path/base64/url accepted by Qwen3TTSTokenizer")
    parser.add_argument("--output-codes", required=True)
    parser.add_argument("--qwen3-tts-repo", default="../Qwen3-TTS")
    parser.add_argument("--device-map", default=None)
    args = parser.parse_args()

    speech_tokenizer_dir = Path(args.speech_tokenizer_dir).expanduser().resolve()
    audio = args.audio
    output_codes = Path(args.output_codes).expanduser().resolve()
    qwen_repo = Path(args.qwen3_tts_repo).expanduser().resolve()

    import torch

    Qwen3TTSTokenizer = _load_qwen_tokenizer(qwen_repo)
    kwargs = {}
    if args.device_map:
        kwargs["device_map"] = args.device_map
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        kwargs["device_map"] = "mps"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(speech_tokenizer_dir), **kwargs)

    enc = tokenizer.encode(audio, return_dict=True)
    audio_codes = enc.audio_codes
    if not audio_codes:
        raise RuntimeError("Speech tokenizer encode returned no audio_codes.")
    codes = audio_codes[0]
    _save_codes(output_codes, codes)

    if codes.dim() == 1:
        frames, groups = int(codes.shape[0]), 1
    else:
        frames, groups = int(codes.shape[0]), int(codes.shape[1])
    print(f"Encoded audio to {frames} frames x {groups} groups at {output_codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
