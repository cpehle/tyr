#!/usr/bin/env python3
"""
Prepare speaker-encoder mel tensor from reference audio for Lean Qwen3-TTS.

This script mirrors upstream Qwen3-TTS mel extraction used by
`extract_speaker_embedding`:
  - load mono waveform
  - resample to 24 kHz (or provided target SR)
  - compute log-mel with librosa mel filters + torch.stft
  - save tensor as shape [1, frames, mel_dim] via torch.save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import torch
from librosa.filters import mel as librosa_mel_fn
from safetensors.torch import save_file


def _dynamic_range_compression_torch(x: torch.Tensor, c: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * c)


def _mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None = None,
) -> torch.Tensor:
    device = y.device
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(device)
    hann_window = torch.hann_window(win_size).to(device)

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = _dynamic_range_compression_torch(mel_spec)
    return mel_spec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Reference audio path")
    parser.add_argument("--output-mel", required=True, help="Output safetensors path (.safetensors)")
    parser.add_argument("--output-frames", required=True, help="Output text file containing mel frame count")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--num-mels", type=int, default=128)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--win-size", type=int, default=1024)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=12000)
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    output_mel = Path(args.output_mel).expanduser().resolve()
    output_frames = Path(args.output_frames).expanduser().resolve()

    wav, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if int(sr) != int(args.target_sr):
        wav = librosa.resample(y=wav, orig_sr=int(sr), target_sr=int(args.target_sr))
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)

    mel = _mel_spectrogram(
        wav_t,
        n_fft=int(args.n_fft),
        num_mels=int(args.num_mels),
        sampling_rate=int(args.target_sr),
        hop_size=int(args.hop_size),
        win_size=int(args.win_size),
        fmin=int(args.fmin),
        fmax=int(args.fmax),
    ).transpose(1, 2)

    output_mel.parent.mkdir(parents=True, exist_ok=True)
    output_frames.parent.mkdir(parents=True, exist_ok=True)
    save_file({"mel": mel.detach().cpu().contiguous()}, str(output_mel))
    output_frames.write_text(f"{int(mel.shape[1])}\n", encoding="utf-8")
    print(f"Prepared mel {tuple(mel.shape)} at {output_mel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
