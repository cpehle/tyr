#!/usr/bin/env python3
"""Compare one dumped MhaH100 raw partial tile against a Torch reference.

This helper is intentionally narrow:

- it consumes existing fixture tensors from `Examples/GPU/RunMhaH100*.lean`
- it consumes dumped diagnostic tensors `diag_dK_tiles.pt` / `diag_dV_tiles.pt`
- it compares one selected `(qBlock, kvBlock)` tile against the per-tile
  backward math used by the current ThunderKittens-style MHA path

Expected fixture contents under `--fixture-dir`:

- `q.pt`
- `k.pt`
- `v.pt`
- `dO.pt`
- `expected_o.pt`   by default, or override with `--out-path`
- `expected_l.pt`   by default, or override with `--l-path`

Supported diagnostic tensor layouts:

1. Canonical 4D tiles:
   `[qBlocks, kvBlocks, tileSize, headDim]`

2. Current flat stack contract:
   `[1, 1, kvBlocks * seq, headDim]`
   reshaped internally using the kernel contract:
   `stack_row = qBlock * kvBlocks + kvBlock`

Reference math for one tile `(qBlock, kvBlock)`:

    scale = 1 / sqrt(head_dim)
    lse_q = -scale * l[qBlock]
    d_vec_q = row_sum(dO_q * O_q)

    scores_t = (K_kv @ Q_q.T) * scale
    P_t      = exp(scores_t - lse_q[None, :])
    dP_t     = V_kv @ dO_q.T
    dS_t     = P_t * (dP_t - d_vec_q[None, :]) * scale

    dV_part  = P_t  @ dO_q
    dK_part  = dS_t @ Q_q

Usage example:

    python3 scripts/gpu/compare_mha_partial_tile.py \
      --fixture-dir data/gpu_fixtures/mha_h100_128x64 \
      --diag-dk /tmp/diag_dK_tiles.pt \
      --diag-dv /tmp/diag_dV_tiles.pt \
      --q-block 0 \
      --kv-block 1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        required=True,
        help="Directory containing q/k/v/dO and expected fixture tensors.",
    )
    parser.add_argument(
        "--diag-dk",
        type=Path,
        required=True,
        help="Path to dumped dK diagnostic tiles.",
    )
    parser.add_argument(
        "--diag-dv",
        type=Path,
        required=True,
        help="Path to dumped dV diagnostic tiles.",
    )
    parser.add_argument(
        "--q-block",
        type=int,
        required=True,
        help="Selected query block index.",
    )
    parser.add_argument(
        "--kv-block",
        type=int,
        required=True,
        help="Selected KV block index.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Optional override for the output tensor used to build d_vec.",
    )
    parser.add_argument(
        "--l-path",
        type=Path,
        default=None,
        help="Optional override for the L tensor used to reconstruct lse.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Attention tile size. The current kernels use 64.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=3e-2,
        help="Relative tolerance for allclose checks.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=3e-2,
        help="Absolute tolerance for allclose checks.",
    )
    parser.add_argument(
        "--save-ref-dir",
        type=Path,
        default=None,
        help="Optional directory to write the computed reference dK/dV tiles.",
    )
    return parser.parse_args()


def _require_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "compare_mha_partial_tile.py requires the Python 'torch' package. "
            "Run it in an environment with PyTorch available."
        ) from exc
    return torch


def _load_tensor(torch: Any, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing tensor file: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if hasattr(torch.jit, "RecursiveScriptModule") and isinstance(
        obj, torch.jit.RecursiveScriptModule
    ):
        state = obj.state_dict()
        tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0].detach().cpu()
    if isinstance(obj, dict):
        tensors = [value for value in obj.values() if isinstance(value, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0].detach().cpu()
    raise TypeError(f"unsupported tensor payload in {path}: {type(obj)!r}")


def _fixture_path(fixture_dir: Path, name: str) -> Path:
    return fixture_dir / f"{name}.pt"


def _canonicalize_l(torch: Any, l_tensor, q_blocks: int, tile_size: int):
    l_tensor = l_tensor.detach().cpu().to(torch.float32).contiguous()
    if tuple(l_tensor.shape) == (q_blocks, tile_size):
        return l_tensor
    if tuple(l_tensor.shape) == (1, 1, q_blocks * tile_size):
        return l_tensor.reshape(q_blocks, tile_size)
    raise ValueError(
        f"unsupported L tensor shape {tuple(l_tensor.shape)}; "
        f"expected {(q_blocks, tile_size)} or {(1, 1, q_blocks * tile_size)}"
    )


def _canonicalize_diag_tiles(
    torch: Any,
    diag,
    *,
    q_blocks: int,
    kv_blocks: int,
    tile_size: int,
    head_dim: int,
    name: str,
):
    diag = diag.detach().cpu().to(torch.float32).contiguous()
    want = (q_blocks, kv_blocks, tile_size, head_dim)
    if tuple(diag.shape) == want:
        return diag
    flat_stack = (1, 1, kv_blocks * q_blocks * tile_size, head_dim)
    if tuple(diag.shape) == flat_stack:
        return diag.reshape(q_blocks, kv_blocks, tile_size, head_dim)
    if tuple(diag.shape) == (q_blocks * kv_blocks, tile_size, head_dim):
        return diag.reshape(q_blocks, kv_blocks, tile_size, head_dim)
    raise ValueError(
        f"unsupported {name} tensor shape {tuple(diag.shape)}; "
        f"expected {want}, {flat_stack}, or {(q_blocks * kv_blocks, tile_size, head_dim)}"
    )


def _select_tile(tensor, q_block: int, kv_block: int):
    return tensor[q_block, kv_block]


def _as_bf16_f32(torch: Any, tensor):
    return tensor.to(torch.bfloat16).to(torch.float32)


def _compute_reference_tiles(
    torch: Any,
    *,
    q,
    k,
    v,
    dO,
    out,
    l_rows,
    q_block: int,
    kv_block: int,
    tile_size: int,
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or dO.ndim != 4 or out.ndim != 4:
        raise ValueError("expected q/k/v/dO/out tensors with rank 4 and shape [B, H, S, D]")
    if q.shape[0] != 1 or q.shape[1] != 1:
        raise ValueError(
            f"this helper currently expects single-batch single-head fixtures, got q shape {tuple(q.shape)}"
        )

    head_dim = int(q.shape[-1])
    scale = 1.0 / math.sqrt(head_dim)

    q_start = q_block * tile_size
    q_stop = q_start + tile_size
    kv_start = kv_block * tile_size
    kv_stop = kv_start + tile_size

    q_tile = q[0, 0, q_start:q_stop, :].to(torch.float32)
    k_tile = k[0, 0, kv_start:kv_stop, :].to(torch.float32)
    v_tile = v[0, 0, kv_start:kv_stop, :].to(torch.float32)
    dO_tile = dO[0, 0, q_start:q_stop, :].to(torch.float32)
    out_tile = out[0, 0, q_start:q_stop, :].to(torch.float32)
    l_tile = l_rows[q_block].to(torch.float32)

    d_vec = (dO_tile * out_tile).sum(dim=1)
    lse = -scale * l_tile

    scores_t = torch.matmul(k_tile, q_tile.transpose(0, 1)) * scale
    probs_t = torch.exp(scores_t - lse.unsqueeze(0))
    dP_t = torch.matmul(v_tile, dO_tile.transpose(0, 1))
    dS_t = probs_t * (dP_t - d_vec.unsqueeze(0)) * scale

    dK_refs = {
        "fp32": torch.matmul(dS_t, q_tile),
        "ds_bf16": torch.matmul(_as_bf16_f32(torch, dS_t), q_tile),
        "q_bf16": torch.matmul(dS_t, _as_bf16_f32(torch, q_tile)),
        "both_bf16": torch.matmul(_as_bf16_f32(torch, dS_t), _as_bf16_f32(torch, q_tile)),
        "dsT_q": torch.matmul(dS_t.transpose(0, 1), q_tile),
        "ds_qT": torch.matmul(dS_t, q_tile.transpose(0, 1)),
        "dsT_qT": torch.matmul(dS_t.transpose(0, 1), q_tile.transpose(0, 1)),
    }
    dV_refs = {
        "fp32": torch.matmul(probs_t, dO_tile),
        "p_bf16": torch.matmul(_as_bf16_f32(torch, probs_t), dO_tile),
        "do_bf16": torch.matmul(probs_t, _as_bf16_f32(torch, dO_tile)),
        "both_bf16": torch.matmul(_as_bf16_f32(torch, probs_t), _as_bf16_f32(torch, dO_tile)),
    }
    return dK_refs, dV_refs


def _compare_tile(torch: Any, actual, reference, *, rtol: float, atol: float):
    diff = (actual - reference).abs()
    return {
        "ok": bool(torch.allclose(actual, reference, rtol=rtol, atol=atol)),
        "mae": float(diff.mean().item()),
        "max": float(diff.max().item()),
    }


def _print_summary(
    *,
    q_block: int,
    kv_block: int,
    stack_row: int,
    dk_stats: dict[str, Any],
    dv_stats: dict[str, Any],
    ) -> None:
    print(
        "mha_h100_partial_tile "
        f"q_block={q_block} "
        f"kv_block={kv_block} "
        f"stack_row={stack_row} "
        f"dk_ok={dk_stats['ok']} "
        f"dk_mae={dk_stats['mae']:.6f} "
        f"dk_max={dk_stats['max']:.6f} "
        f"dv_ok={dv_stats['ok']} "
        f"dv_mae={dv_stats['mae']:.6f} "
        f"dv_max={dv_stats['max']:.6f}"
    )


def _print_hypotheses(prefix: str, stats_by_name: dict[str, dict[str, Any]]) -> None:
    for name, stats in stats_by_name.items():
        print(
            f"{prefix}_hypothesis "
            f"name={name} "
            f"ok={stats['ok']} "
            f"mae={stats['mae']:.6f} "
            f"max={stats['max']:.6f}"
        )


def _save_reference_tiles(torch: Any, save_dir: Path, q_block: int, kv_block: int, dK_ref, dV_ref) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    dk_path = save_dir / f"ref_dK_q{q_block}_kv{kv_block}.pt"
    dv_path = save_dir / f"ref_dV_q{q_block}_kv{kv_block}.pt"
    torch.save(dK_ref, dk_path)
    torch.save(dV_ref, dv_path)
    print(f"saved_ref_dk={dk_path}")
    print(f"saved_ref_dv={dv_path}")


def run(args: argparse.Namespace) -> int:
    torch = _require_torch()

    q = _load_tensor(torch, _fixture_path(args.fixture_dir, "q"))
    k = _load_tensor(torch, _fixture_path(args.fixture_dir, "k"))
    v = _load_tensor(torch, _fixture_path(args.fixture_dir, "v"))
    dO = _load_tensor(torch, _fixture_path(args.fixture_dir, "dO"))
    out_path = args.out_path or _fixture_path(args.fixture_dir, "expected_o")
    l_path = args.l_path or _fixture_path(args.fixture_dir, "expected_l")
    out = _load_tensor(torch, out_path)
    l_tensor = _load_tensor(torch, l_path)
    dK_diag = _load_tensor(torch, args.diag_dk)
    dV_diag = _load_tensor(torch, args.diag_dv)

    if q.ndim != 4:
        raise ValueError(f"expected q fixture with rank 4, got {tuple(q.shape)}")

    seq = int(q.shape[2])
    head_dim = int(q.shape[3])
    tile_size = int(args.tile_size)
    if seq % tile_size != 0:
        raise ValueError(f"sequence length {seq} is not divisible by tile size {tile_size}")

    q_blocks = seq // tile_size
    kv_seq = int(k.shape[2])
    if kv_seq % tile_size != 0:
        raise ValueError(f"KV sequence length {kv_seq} is not divisible by tile size {tile_size}")
    kv_blocks = kv_seq // tile_size

    if not (0 <= args.q_block < q_blocks):
        raise ValueError(f"q_block {args.q_block} is out of range [0, {q_blocks})")
    if not (0 <= args.kv_block < kv_blocks):
        raise ValueError(f"kv_block {args.kv_block} is out of range [0, {kv_blocks})")

    l_rows = _canonicalize_l(torch, l_tensor, q_blocks, tile_size)
    dK_tiles = _canonicalize_diag_tiles(
        torch,
        dK_diag,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        tile_size=tile_size,
        head_dim=head_dim,
        name="diag_dK_tiles",
    )
    dV_tiles = _canonicalize_diag_tiles(
        torch,
        dV_diag,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
        tile_size=tile_size,
        head_dim=head_dim,
        name="diag_dV_tiles",
    )

    dK_refs, dV_refs = _compute_reference_tiles(
        torch,
        q=q,
        k=k,
        v=v,
        dO=dO,
        out=out,
        l_rows=l_rows,
        q_block=args.q_block,
        kv_block=args.kv_block,
        tile_size=tile_size,
    )

    dK_actual = _select_tile(dK_tiles, args.q_block, args.kv_block)
    dV_actual = _select_tile(dV_tiles, args.q_block, args.kv_block)

    dk_stats_by_name = {
        name: _compare_tile(torch, dK_actual, ref, rtol=args.rtol, atol=args.atol)
        for name, ref in dK_refs.items()
    }
    dv_stats_by_name = {
        name: _compare_tile(torch, dV_actual, ref, rtol=args.rtol, atol=args.atol)
        for name, ref in dV_refs.items()
    }
    dk_stats = dk_stats_by_name["fp32"]
    dv_stats = dv_stats_by_name["fp32"]
    stack_row = args.q_block * kv_blocks + args.kv_block

    print(
        "fixture "
        f"seq={seq} "
        f"head_dim={head_dim} "
        f"tile_size={tile_size} "
        f"q_blocks={q_blocks} "
        f"kv_blocks={kv_blocks} "
        f"out_path={out_path} "
        f"l_path={l_path}"
    )
    _print_summary(
        q_block=args.q_block,
        kv_block=args.kv_block,
        stack_row=stack_row,
        dk_stats=dk_stats,
        dv_stats=dv_stats,
    )
    _print_hypotheses("dk", dk_stats_by_name)
    _print_hypotheses("dv", dv_stats_by_name)

    if args.save_ref_dir is not None:
        _save_reference_tiles(
            torch,
            args.save_ref_dir,
            args.q_block,
            args.kv_block,
            dK_refs["fp32"],
            dV_refs["fp32"],
        )

    return 0 if dk_stats["ok"] and dv_stats["ok"] else 1


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
