import Tyr.GPU.Kernels.GemmCommon

/-!
  Tyr/GPU/Kernels/Bf16Gemm.lean

  BF16 GEMM counterparts for the vendored ThunderKittens GEMM catalog.

  - `tkH100Bf16GemmFwd` is the canonical H100/Hopper BF16 surface aligned with
    `kernels/gemm/bf16_h100/bf16_h100_gemm.cu`.
  - `tkB200Bf16GemmFwd` is the Blackwell/B200 surface aligned with
    `kernels/gemm/bf16_b200/bf16_b200_gemm.cu`.
-/

namespace Tyr.GPU.Kernels.Bf16Gemm

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev h100TileM : Nat := 128
private abbrev h100TileK : Nat := 64
private abbrev h100TileN : Nat := 256
private abbrev h100KBlocks : Nat := 4

private abbrev b200TileM : Nat := 256
private abbrev b200TileK : Nat := 64
private abbrev b200TileN : Nat := 256
private abbrev b200KBlocks : Nat := 4

private def h100Bf16Accumulator
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 h100TileM h100TileN × RTileCoord) :=
  GemmCommon.tiledAccumulator
    (tileM := h100TileM)
    (tileK := h100TileK)
    (tileN := h100TileN)
    (kBlocks := h100KBlocks)
    "=== H100 BF16 GEMM ==="
    "ThunderKittens bf16_h100 producer/consumer tile, expressed as a single CTA-local tiled mainloop"
    aPtr bPtr m n k

private def b200Bf16Accumulator
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 b200TileM b200TileN × RTileCoord) :=
  GemmCommon.tiledAccumulator
    (tileM := b200TileM)
    (tileK := b200TileK)
    (tileN := b200TileN)
    (kBlocks := b200KBlocks)
    "=== B200 BF16 GEMM ==="
    "ThunderKittens bf16_b200 producer/consumer cluster-TMEM surface represented as a typed Blackwell-sized tiled mainloop"
    aPtr bPtr m n k

/-- Canonical H100 BF16 GEMM surface matching the tile geometry used by the
vendored `bf16_h100` kernel family. -/
@[gpu_kernel .SM90]
def tkH100Bf16GemmFwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← h100Bf16Accumulator aPtr bPtr m n k
  GemmCommon.storeConvertedTile cPtr coord accum

/-- Blackwell/B200 BF16 surface aligned with `bf16_b200`.

This keeps the Blackwell tile geometry and accumulator contract explicit while
flattening the source's cluster/TMEM worker choreography into one typed
CTA-local tiled mainloop plus typed epilogue. -/
@[gpu_kernel .SM100]
def tkB200Bf16GemmFwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← b200Bf16Accumulator aPtr bPtr m n k
  GemmCommon.storeConvertedTile cPtr coord accum

end Tyr.GPU.Kernels.Bf16Gemm
