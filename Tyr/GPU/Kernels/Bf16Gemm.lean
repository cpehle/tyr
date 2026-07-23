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

-- GB10 has a 48 KiB per-CTA shared-memory limit. Keep the complete A, B,
-- and output staging footprint below that limit for the small-GEMM route.
private abbrev gb10TileM : Nat := 64
private abbrev gb10TileK : Nat := 64
private abbrev gb10TileN : Nat := 64
private abbrev gb10KBlocks : Nat := 1

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

private def gb10Bf16Accumulator (kBlocks : Nat)
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 gb10TileM gb10TileN × RTileCoord) :=
  GemmCommon.tiledAccumulator
    (tileM := gb10TileM)
    (tileK := gb10TileK)
    (tileN := gb10TileN)
    (kBlocks := kBlocks)
    "=== GB10 BF16 GEMM ==="
    "GB10 64x64x64 CTA tile sized for the physical 48 KiB shared-memory limit"
    aPtr bPtr m n k

/-- Single-buffer TMA-lookahead mainloop for fixed-K GB10 GEMMs. Once a tile
    has been copied from shared memory into warp registers, the next A/B tile
    can overwrite that shared stage concurrently with the current warp MMA. -/
private def gb10Bf16AccumulatorPipelined (kBlocks : Nat)
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 gb10TileM gb10TileN × RTileCoord) := do
  let _ := (m, n, k)
  let coord ← blockCoord2D
  let a ← allocRT .BFloat16 gb10TileM gb10TileK
  let b ← allocRT .BFloat16 gb10TileN gb10TileK
  let accum ← zeroRT .Float32 gb10TileM gb10TileN
  let aShared ← allocST .BFloat16 gb10TileM gb10TileK
  let bShared ← allocST .BFloat16 gb10TileN gb10TileK
  let sem ← allocSemaphore
  initSemaphore sem 1
  let tileBytes := gb10TileM * gb10TileK * GpuFloat.bytes .BFloat16
  let issue (kBlock : Nat) : KernelM Unit := do
    let kBlockVal ← constIntVal kBlock s!"gemm_k_block_{kBlock}"
    let aCoord := coord.withCol kBlockVal.id
    let bCoord := (coord.withRow coord.c).withCol kBlockVal.id
    expectBytes sem (2 * tileBytes)
    loadGlobalAsync aShared aPtr aCoord sem.id
    loadGlobalAsync bShared bPtr bCoord sem.id
  issue 0
  for kBlock in List.range kBlocks do
    let phase ← constIntVal (kBlock % 2) s!"gemm_phase_{kBlock}"
    waitSemaphorePhaseVal sem phase
    load a aShared
    load b bShared
    if kBlock + 1 < kBlocks then
      issue (kBlock + 1)
    mmaT accum a b accum
  pure (accum, coord)

/-- Four-warp GB10 mainloop. Each warp owns 16 distinct rows of a shared
    64x64 output tile, while all warps reuse the same staged A/B K tile. -/
private def gb10Bf16AccumulatorWarp4 (kBlocks : Nat)
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 16 gb10TileN × RTileCoord) := do
  let _ := (m, n, k)
  let coord ← blockCoord2D
  let a ← allocRT .BFloat16 16 gb10TileK
  let b ← allocRT .BFloat16 gb10TileN gb10TileK
  let accum ← zeroRT .Float32 16 gb10TileN
  let aShared0 ← allocST .BFloat16 gb10TileM gb10TileK
  let aShared1 ← allocST .BFloat16 gb10TileM gb10TileK
  let bShared0 ← allocST .BFloat16 gb10TileN gb10TileK
  let bShared1 ← allocST .BFloat16 gb10TileN gb10TileK
  let sem0 ← allocSemaphore
  let sem1 ← allocSemaphore
  initSemaphore sem0 1
  initSemaphore sem1 1
  blockSync
  let warpId ← getWarpId "gemm_warp4_warp"
  let zero ← constIntVal 0 "gemm_warp4_zero"
  let warpZero ← scalarEq warpId zero "gemm_warp4_is_zero"
  let tileBytes := gb10TileM * gb10TileK * GpuFloat.bytes .BFloat16
  let issue (aShared : ST GpuFloat.BFloat16 gb10TileM gb10TileK)
      (bShared : ST GpuFloat.BFloat16 gb10TileN gb10TileK)
      (sem : Semaphore) (kBlock : Nat) : KernelM Unit := do
    let kBlockVal ← constIntVal kBlock s!"gemm_warp4_k_{kBlock}"
    let aCoord := coord.withCol kBlockVal.id
    let bCoord := (coord.withRow coord.c).withCol kBlockVal.id
    expectBytes sem (2 * tileBytes)
    loadGlobalAsync aShared aPtr aCoord sem.id
    loadGlobalAsync bShared bPtr bCoord sem.id
  issue aShared0 bShared0 sem0 0
  if kBlocks > 1 then issue aShared1 bShared1 sem1 1
  for kBlock in List.range kBlocks do
    let phase ← constIntVal ((kBlock / 2) % 2) s!"gemm_warp4_phase_{kBlock}"
    emitIf warpZero.id do
      if kBlock % 2 == 0 then
        waitSemaphorePhaseVal sem0 phase
      else
        waitSemaphorePhaseVal sem1 phase
    blockSync
    if kBlock % 2 == 0 then
      warpgroupLoad a aShared0
      load b bShared0
    else
      warpgroupLoad a aShared1
      load b bShared1
    blockSync
    if kBlock + 2 < kBlocks then
      if kBlock % 2 == 0 then
        issue aShared0 bShared0 sem0 (kBlock + 2)
      else
        issue aShared1 bShared1 sem1 (kBlock + 2)
    mmaT accum a b accum
  let inputRow : KVal UInt32 := ⟨coord.r, "gemm_warp4_input_row"⟩
  let four ← constIntVal 4 "gemm_warp4_four"
  let outputRow ← scalarMulVal inputRow four "gemm_warp4_output_row"
  pure (accum, coord.withRow outputRow.id)

/-- Compact two-stage four-warp mainloop for runtime K multiples of 64. -/
private def gb10Bf16AccumulatorWarp4Runtime (kBlocks : KVal UInt32)
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM (RT GpuFloat.Float32 16 gb10TileN × RTileCoord) := do
  let _ := (m, n, k)
  let coord ← blockCoord2D
  let a ← allocRT .BFloat16 16 gb10TileK
  let b ← allocRT .BFloat16 gb10TileN gb10TileK
  let accum ← zeroRT .Float32 16 gb10TileN
  let aShared0 ← allocST .BFloat16 gb10TileM gb10TileK
  let aShared1 ← allocST .BFloat16 gb10TileM gb10TileK
  let bShared0 ← allocST .BFloat16 gb10TileN gb10TileK
  let bShared1 ← allocST .BFloat16 gb10TileN gb10TileK
  let sem0 ← allocSemaphore
  let sem1 ← allocSemaphore
  initSemaphore sem0 1
  initSemaphore sem1 1
  blockSync
  let warpId ← getWarpId "gemm_runtime_warp"
  let zero ← constIntVal 0 "gemm_runtime_zero"
  let one ← constIntVal 1 "gemm_runtime_one"
  let two ← constIntVal 2 "gemm_runtime_two"
  let four ← constIntVal 4 "gemm_runtime_four"
  let warpZero ← scalarEq warpId zero "gemm_runtime_is_warp_zero"
  let tileBytes := gb10TileM * gb10TileK * GpuFloat.bytes .BFloat16
  let issue (aShared : ST GpuFloat.BFloat16 gb10TileM gb10TileK)
      (bShared : ST GpuFloat.BFloat16 gb10TileN gb10TileK)
      (sem : Semaphore) (kBlock : KVal UInt32) : KernelM Unit := do
    let aCoord := coord.withCol kBlock.id
    let bCoord := (coord.withRow coord.c).withCol kBlock.id
    expectBytes sem (2 * tileBytes)
    loadGlobalAsync aShared aPtr aCoord sem.id
    loadGlobalAsync bShared bPtr bCoord sem.id
  issue aShared0 bShared0 sem0 zero
  issue aShared1 bShared1 sem1 one
  for kBlock in kvrange 0 kBlocks do
    let kBlockVal : KVal UInt32 := ⟨kBlock.id, "gemm_runtime_k_block"⟩
    let stage ← scalarMod kBlockVal two "gemm_runtime_stage"
    let half ← scalarDivVal kBlockVal two "gemm_runtime_half"
    let phase ← scalarMod half two "gemm_runtime_phase"
    let isStage0 ← scalarEq stage zero "gemm_runtime_is_stage0"
    emitIf warpZero.id do
      ifThenElse isStage0
        (waitSemaphorePhaseVal sem0 phase)
        (waitSemaphorePhaseVal sem1 phase)
    blockSync
    ifThenElse isStage0
      (do warpgroupLoad a aShared0; load b bShared0)
      (do warpgroupLoad a aShared1; load b bShared1)
    blockSync
    let nextBlock ← scalarAddVal kBlockVal two "gemm_runtime_next_block"
    let hasNext ← scalarLt nextBlock kBlocks "gemm_runtime_has_next"
    emitIf hasNext.id do
      ifThenElse isStage0
        (issue aShared0 bShared0 sem0 nextBlock)
        (issue aShared1 bShared1 sem1 nextBlock)
    mmaT accum a b accum
  let inputRow : KVal UInt32 := ⟨coord.r, "gemm_runtime_input_row"⟩
  let outputRow ← scalarMulVal inputRow four "gemm_runtime_output_row"
  pure (accum, coord.withRow outputRow.id)

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

/-- GB10 small BF16 GEMM route. The physical target is supplied by the native
    build as sm_121; this kernel deliberately uses only the portable tiled MMA
    surface and does not claim B200 TMEM or cluster capabilities. -/
@[gpu_kernel .SM100]
def tkGB10Bf16GemmFwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  setFamily .Blackwell
  let (accum, coord) ← gb10Bf16Accumulator gb10KBlocks aPtr bPtr m n k
  GemmCommon.storeConvertedTile cPtr coord accum

/-- GB10 BF16 GEMM specialization for K=256 (four 64-wide K tiles). -/
@[gpu_kernel .SM100]
def tkGB10Bf16GemmK256Fwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  setFamily .Blackwell
  let (accum, coord) ← gb10Bf16AccumulatorWarp4 4 aPtr bPtr m n k
  let out ← allocRT .BFloat16 16 gb10TileN
  convert out accum
  warpgroupStoreGlobal cPtr out coord

/-- GB10 BF16 GEMM specialization for K=1024 (sixteen 64-wide K tiles). -/
@[gpu_kernel .SM100]
def tkGB10Bf16GemmK1024Fwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  setFamily .Blackwell
  let (accum, coord) ← gb10Bf16AccumulatorWarp4 16 aPtr bPtr m n k
  let out ← allocRT .BFloat16 16 gb10TileN
  convert out accum
  warpgroupStoreGlobal cPtr out coord

/-- GB10 BF16 GEMM route for runtime K multiples of 64 (K >= 128). -/
@[gpu_kernel .SM100]
def tkGB10Bf16GemmKRuntimeFwd
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.BFloat16)
    (m : KVal UInt64)
    (n : KVal UInt64)
    (k : KVal UInt64)
    : KernelM Unit := do
  setFamily .Blackwell
  let k32 : KVal UInt32 ← castScalar .UInt32 k "gemm_runtime_k32"
  let tileK ← constIntVal 64 "gemm_runtime_tile_k"
  let kBlocks ← scalarDivVal k32 tileK "gemm_runtime_k_blocks"
  let (accum, coord) ← gb10Bf16AccumulatorWarp4Runtime kBlocks aPtr bPtr m n k
  let out ← allocRT .BFloat16 16 gb10TileN
  convert out accum
  warpgroupStoreGlobal cPtr out coord

end Tyr.GPU.Kernels.Bf16Gemm
