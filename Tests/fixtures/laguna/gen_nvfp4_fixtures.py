#!/usr/bin/env python
"""Generate NVFP4 dequantization ground-truth fixtures for Tests/RunLagunaNvFp4.

Synthesizes random packed nibbles + random positive e4m3 scales + random
global scales, computes the reference dequantization in FP32
(`e2m1 * (scale / global)` in the same op order as the Lean implementation),
rounds to BF16, and saves everything to nvfp4.safetensors.

Run with: .venv-gpu/bin/python Tests/fixtures/laguna/gen_nvfp4_fixtures.py
"""
import os

import torch
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvfp4.safetensors")

E2M1_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def e2m1_decode(nibbles: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 nibbles (int tensor, values 0..15) to float32."""
    n = nibbles.to(torch.int32)
    sign = (n >> 3) & 1
    mag = E2M1_MAGS[n & 7]
    return torch.where(sign.bool(), -mag, mag)


def reference_dequant(packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize [.., out, in/2] U8 + [.., out, in/16] F8 + [.., 1, 1] F32 global -> F32 [.., out, in]."""
    b = packed.to(torch.int32)
    lo = e2m1_decode(b & 0xF)          # element 2i
    hi = e2m1_decode((b >> 4) & 0xF)   # element 2i+1
    vals = torch.stack((lo, hi), dim=-1).flatten(-2)  # [.., out, in]
    scale_full = (scales.to(torch.float32) / global_scale).repeat_interleave(16, dim=-1)
    return vals * scale_full


def make_bank(shape, seed):
    """shape = (lead, out, ins) -> dict of packed/scales/global/expected tensors."""
    lead, out, ins = shape
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (lead, out, ins // 2), generator=g, dtype=torch.uint8)
    scales = (torch.rand((lead, out, ins // 16), generator=g) * 0.48 + 0.02).to(torch.float8_e4m3fn)
    global_scale = torch.rand((lead,), generator=g) * 0.4 + 0.8  # [0.8, 1.2)
    expected = reference_dequant(packed, scales, global_scale.view(lead, 1, 1)).to(torch.bfloat16)
    return packed, scales, global_scale, expected


def main():
    torch.manual_seed(0)
    # Single matrix: out=64, in=128.
    m_packed, m_scales, m_global, m_expected = make_bank((1, 64, 128), seed=1234)
    # Stacked bank: E=4, out=32, in=64.
    b_packed, b_scales, b_global, b_expected = make_bank((4, 32, 64), seed=5678)

    tensors = {
        "m_packed": m_packed.squeeze(0).contiguous(),          # [64, 64] U8
        "m_scales": m_scales.squeeze(0).contiguous(),          # [64, 8] F8_E4M3
        "m_global": m_global.contiguous(),                     # [1] F32
        "m_expected": m_expected.squeeze(0).contiguous(),      # [64, 128] BF16
        "b_packed": b_packed.contiguous(),                     # [4, 32, 32] U8
        "b_scales": b_scales.contiguous(),                     # [4, 32, 4] F8_E4M3
        "b_global": b_global.contiguous(),                     # [4] F32
        "b_expected": b_expected.contiguous(),                 # [4, 32, 64] BF16
    }
    save_file(tensors, OUT)
    print(f"wrote {OUT}")
    for k, v in tensors.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
