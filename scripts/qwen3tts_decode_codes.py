#!/usr/bin/env python3
"""
Decode Qwen3-TTS codec tokens (12Hz speech tokenizer) into a WAV file.

Input `codes` format:
  One frame per line, whitespace-separated codebook token IDs.
  Example with 4 code groups:
    12 98 44 777
    13 97 41 771
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path


def _load_codes(path: Path, num_code_groups: int) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != num_code_groups:
                raise ValueError(
                    f"{path}:{line_no}: expected {num_code_groups} columns, got {len(parts)}"
                )
            rows.append([int(p) for p in parts])

    if not rows:
        raise ValueError(f"No codec rows found in {path}")
    return rows


def _save_wav(path: Path, wav, sample_rate: int) -> None:
    from array import array

    pcm = array("h")
    for x in wav:
        xf = float(x)
        if xf > 1.0:
            xf = 1.0
        elif xf < -1.0:
            xf = -1.0
        pcm.append(int(xf * 32767.0))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _load_qwen_tokenizer(repo_path: Path):
    try:
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

        return Qwen3TTSTokenizer
    except Exception:
        sys.path.insert(0, str(repo_path))
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

        return Qwen3TTSTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speech-tokenizer-dir", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--num-code-groups", required=True, type=int)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--qwen3-tts-repo", default="../Qwen3-TTS")
    parser.add_argument("--device-map", default=None)
    args = parser.parse_args()

    speech_tokenizer_dir = Path(args.speech_tokenizer_dir).expanduser().resolve()
    codes_path = Path(args.codes).expanduser().resolve()
    output_wav = Path(args.output_wav).expanduser().resolve()
    qwen_repo = Path(args.qwen3_tts_repo).expanduser().resolve()

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Missing Python dependency: torch") from exc

    codes_rows = _load_codes(codes_path, args.num_code_groups)
    codes = torch.tensor(codes_rows, dtype=torch.long)

    Qwen3TTSTokenizer = _load_qwen_tokenizer(qwen_repo)

    kwargs = {}
    if args.device_map:
        kwargs["device_map"] = args.device_map
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        kwargs["device_map"] = "mps"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(speech_tokenizer_dir), **kwargs)

    wavs, sample_rate = tokenizer.decode({"audio_codes": [codes]})
    if not wavs:
        raise RuntimeError("Tokenizer decode returned no waveforms.")
    wav = wavs[0]

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    _save_wav(output_wav, wav, sample_rate)
    print(f"Decoded {len(codes_rows)} frames to {output_wav} @ {sample_rate} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
