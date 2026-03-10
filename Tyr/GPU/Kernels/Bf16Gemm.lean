import Tyr.GPU.Kernels.GemmCommon

/-!
  Tyr/GPU/Kernels/Bf16Gemm.lean

  BF16 GEMM counterparts for the vendored ThunderKittens GEMM catalog.

  - `tkH100Bf16GemmFwd` is the canonical H100/Hopper BF16 surface aligned with
    `kernels/gemm/bf16_h100/bf16_h100_gemm.cu`.
  - `tkB200Bf16GemmCompatFwd` is the Blackwell-oriented compatibility surface
    aligned with `kernels/gemm/bf16_b200/bf16_b200_gemm.cu`, but kept honest
    about the current DSL's lack of explicit TMEM, cluster scheduler, and
    overlap controls.
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

private def b200Bf16CompatAccumulator
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
    "=== B200 BF16 GEMM (compatibility) ==="
    "Representative Blackwell bf16_b200 tile from the vendored configs, flattened to a CTA-local compatibility surface"
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

/-- Blackwell/B200 BF16 compatibility surface aligned with `bf16_b200`.

The vendored kernel relies on cluster scheduling, TMEM, and explicit epilogue
pipeline controls that the current Lean DSL does not model, so this surface
keeps the source's 256x256x64-style block geometry while staying explicit
about its compatibility status. -/
@[gpu_kernel .SM100]
def tkB200Bf16GemmCompatFwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  let (accum, coord) ← b200Bf16CompatAccumulator aPtr bPtr m n k
  GemmCommon.storeConvertedTile cPtr coord accum

end Tyr.GPU.Kernels.Bf16Gemm
