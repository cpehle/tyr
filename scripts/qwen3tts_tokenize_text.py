#!/usr/bin/env python3
"""
Tokenize text for Qwen3-TTS models and write token IDs to a text file.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ids = tok.encode(
        args.text,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_len,
    )

    if not ids:
        raise RuntimeError("Tokenizer produced no token IDs.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(" ".join(str(int(x)) for x in ids) + "\n", encoding="utf-8")
    print(f"Tokenized {len(ids)} ids to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
