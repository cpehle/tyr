#!/usr/bin/env python3
"""Convert Kokoro PyTorch checkpoints and voicepacks into safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def normalize_state_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def convert_checkpoint(src: Path, dst: Path) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict checkpoint, got {type(payload).__name__}")

    tensors: dict[str, torch.Tensor] = {}
    metadata = {"source_format": "pytorch", "source_path": str(src)}

    for top_key, state_dict in payload.items():
        if not isinstance(state_dict, dict):
            continue
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            flat_name = f"{top_key}.{normalize_state_key(name)}"
            tensors[flat_name] = tensor.detach().cpu().contiguous()

    if not tensors:
        raise RuntimeError(f"no tensors found in checkpoint {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst), metadata=metadata)


def convert_voice(src: Path, dst: Path) -> None:
    tensor = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected tensor voicepack, got {type(tensor).__name__}")
    metadata = {"source_format": "pytorch", "source_path": str(src)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file({"voice": tensor.detach().cpu().contiguous()}, str(dst), metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="PyTorch Kokoro checkpoint (.pth)")
    parser.add_argument("--checkpoint-out", type=Path, help="Output model safetensors path")
    parser.add_argument("--voice", type=Path, help="PyTorch voicepack (.pt)")
    parser.add_argument("--voice-out", type=Path, help="Output voice safetensors path")
    args = parser.parse_args()

    did_work = False

    if args.checkpoint or args.checkpoint_out:
        if not args.checkpoint or not args.checkpoint_out:
            parser.error("--checkpoint and --checkpoint-out must be provided together")
        convert_checkpoint(args.checkpoint, args.checkpoint_out)
        did_work = True

    if args.voice or args.voice_out:
        if not args.voice or not args.voice_out:
            parser.error("--voice and --voice-out must be provided together")
        convert_voice(args.voice, args.voice_out)
        did_work = True

    if not did_work:
        parser.error("nothing to convert")


if __name__ == "__main__":
    main()
