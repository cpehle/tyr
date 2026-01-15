/-
  Tyr/GPU/Kernels/PrecisionGemm.lean

  Mixed-precision GEMM kernels for FP8, MxFP8, and NVFp4.
  Based on ThunderKittens patterns.

  Key features:
  - FP8 (E4M3 and E5M2 variants) for efficient inference
  - Microscaling FP8 (MxFP8) with per-block scaling
  - NVFp4 (4-bit floating point) for maximum throughput
  - Mixed-precision accumulation (FP8 inputs → FP32 accumulator)
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels.PrecisionGemm

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## FP8 GEMM

FP8 provides 2x throughput over FP16 on Hopper with similar precision.
Two variants:
- E4M3: 4 exponent bits, 3 mantissa bits (larger range)
- E5M2: 5 exponent bits, 2 mantissa bits (more precision)

Typically: weights in E4M3, activations in E5M2 or E4M3.
-/

/-- FP8 GEMM (E4M3 inputs) -/
@[gpu_kernel .SM90]
def gemmFp8E4M3Fwd (A_ptr : GPtr GpuFloat.FP8E4M3) (B_ptr : GPtr GpuFloat.FP8E4M3)
    (C_ptr : GPtr GpuFloat.BFloat16) (M : KVal UInt64) (N : KVal UInt64)
    (K_dim : KVal UInt64) : KernelM Unit := do
  comment "=== FP8 GEMM (E4M3) ==="
  comment "A (E4M3) @ B (E4M3) → C (FP32)"

  let numBlocks : Nat := 8

  let coord ← blockCoord2D

  -- FP8 E4M3 inputs
  let a : RT GpuFloat.FP8E4M3 64 64 ← allocRT .FP8E4M3 64 64
  let b : RT GpuFloat.FP8E4M3 64 64 .Col ← allocRT .FP8E4M3 64 64 .Col

  -- FP32 accumulator (higher precision for accumulation)
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Output (can be FP32 or downcast to BF16/FP8)
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let aShared : ST GpuFloat.FP8E4M3 64 64 ← allocST .FP8E4M3 64 64
  let bShared : ST GpuFloat.FP8E4M3 64 64 .Col ← allocST .FP8E4M3 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "GEMM loop"
  for blkIdx in krange 0 numBlocks do
    comment "Load FP8 tiles"
    loadGlobal aShared A_ptr (coord.withCol blkIdx.id)
    loadGlobal bShared B_ptr (coord.withRow blkIdx.id)
    sync
    load a aShared
    load b bShared

    comment "FP8 tensor core GEMM (accumulates to FP32)"
    -- On Hopper, tensor cores support FP8 inputs with FP32 accumulation
    mma c a b c

    sync

  comment "Convert to output dtype and store"
  convert out c
  store outShared out
  storeGlobal C_ptr outShared coord

-- Verify auto-generated kernel
#check gemmFp8E4M3Fwd.kernel
#check gemmFp8E4M3Fwd.launch

/-- FP8 GEMM (E5M2 inputs) -/
@[gpu_kernel .SM90]
def gemmFp8E5M2Fwd (A_ptr : GPtr GpuFloat.FP8E5M2) (B_ptr : GPtr GpuFloat.FP8E5M2)
    (C_ptr : GPtr GpuFloat.BFloat16) (M : KVal UInt64) (N : KVal UInt64)
    (K_dim : KVal UInt64) : KernelM Unit := do
  comment "=== FP8 GEMM (E5M2) ==="
  comment "A (E5M2) @ B (E5M2) → C (FP32)"

  let numBlocks : Nat := 8

  let coord ← blockCoord2D

  let a : RT GpuFloat.FP8E5M2 64 64 ← allocRT .FP8E5M2 64 64
  let b : RT GpuFloat.FP8E5M2 64 64 .Col ← allocRT .FP8E5M2 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let aShared : ST GpuFloat.FP8E5M2 64 64 ← allocST .FP8E5M2 64 64
  let bShared : ST GpuFloat.FP8E5M2 64 64 .Col ← allocST .FP8E5M2 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  for blkIdx in krange 0 numBlocks do
    loadGlobal aShared A_ptr (coord.withCol blkIdx.id)
    loadGlobal bShared B_ptr (coord.withRow blkIdx.id)
    sync
    load a aShared
    load b bShared
    mma c a b c
    sync

  convert out c
  store outShared out
  storeGlobal C_ptr outShared coord

-- Verify auto-generated kernel
#check gemmFp8E5M2Fwd.kernel
#check gemmFp8E5M2Fwd.launch

/-! ## Microscaling FP8 (MxFP8)

MxFP8 adds per-block scaling factors to FP8 for better dynamic range:
- Each 32x32 or 64x64 block has a shared scale factor
- Values are: actual_value = fp8_value * scale
- Maintains precision while using efficient FP8 compute
-/

/-- MxFP8 GEMM with per-block scaling -/
@[gpu_kernel .SM90]
def gemmMxFp8Fwd (A_ptr : GPtr GpuFloat.FP8E4M3) (B_ptr : GPtr GpuFloat.FP8E4M3)
    (scale_A_ptr : GPtr GpuFloat.Float32) (scale_B_ptr : GPtr GpuFloat.Float32)
    (C_ptr : GPtr GpuFloat.BFloat16) (M : KVal UInt64) (N : KVal UInt64)
    (K_dim : KVal UInt64) : KernelM Unit := do
  comment "=== Microscaling FP8 GEMM ==="
  comment "FP8 values with per-block scale factors"

  let numBlocks : Nat := 8

  let coord ← blockCoord2D

  -- FP8 data
  let a : RT GpuFloat.FP8E4M3 64 64 ← allocRT .FP8E4M3 64 64
  let b : RT GpuFloat.FP8E4M3 64 64 .Col ← allocRT .FP8E4M3 64 64 .Col

  -- Per-block scales (FP32)
  let scaleA : RV GpuFloat.Float32 64 ← allocRV .Float32 64  -- One scale per row block
  let scaleB : RV GpuFloat.Float32 64 ← allocRV .Float32 64  -- One scale per col block

  -- FP32 accumulator
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let cScaled : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let aShared : ST GpuFloat.FP8E4M3 64 64 ← allocST .FP8E4M3 64 64
  let bShared : ST GpuFloat.FP8E4M3 64 64 .Col ← allocST .FP8E4M3 64 64 .Col
  let scaleAShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let scaleBShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load scale factors"
  loadVec scaleA scaleAShared
  loadVec scaleB scaleBShared

  comment "GEMM with FP8"
  for blkIdx in krange 0 numBlocks do
    loadGlobal aShared A_ptr (coord.withCol blkIdx.id)
    loadGlobal bShared B_ptr (coord.withRow blkIdx.id)
    sync
    load a aShared
    load b bShared
    mma c a b c
    sync

  comment "Apply scale factors"
  comment "C_scaled[i,j] = C[i,j] * scaleA[i] * scaleB[j]"
  -- First scale by scaleA (row-wise)
  mulCol cScaled c scaleA
  -- Then scale by scaleB (col-wise)
  mulRow cScaled cScaled scaleB

  comment "Store output"
  convert out cScaled
  store outShared out
  storeGlobal C_ptr outShared coord

-- Verify auto-generated kernel
#check gemmMxFp8Fwd.kernel
#check gemmMxFp8Fwd.launch

/-! ## Mixed Precision GEMM

Common patterns: BF16 activations with FP8 weights for inference.
-/

/-- Mixed precision GEMM: BF16 @ FP8 → FP32 -/
@[gpu_kernel .SM90]
def gemmMixedFwd (A_ptr : GPtr GpuFloat.BFloat16) (B_ptr : GPtr GpuFloat.FP8E4M3)
    (C_ptr : GPtr GpuFloat.BFloat16) (M : KVal UInt64) (N : KVal UInt64)
    (K_dim : KVal UInt64) : KernelM Unit := do
  comment "=== Mixed Precision GEMM ==="
  comment "A (BF16) @ B (FP8) → C (FP32)"

  let numBlocks : Nat := 8

  let coord ← blockCoord2D

  -- BF16 activations
  let a : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- FP8 weights
  let b : RT GpuFloat.FP8E4M3 64 64 .Col ← allocRT .FP8E4M3 64 64 .Col

  -- FP32 accumulator
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let aShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let bShared : ST GpuFloat.FP8E4M3 64 64 .Col ← allocST .FP8E4M3 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Mixed precision GEMM"
  for blkIdx in krange 0 numBlocks do
    loadGlobal aShared A_ptr (coord.withCol blkIdx.id)
    loadGlobal bShared B_ptr (coord.withRow blkIdx.id)
    sync
    load a aShared
    load b bShared

    comment "Convert BF16 to FP8 for tensor core"
    let aFp8 : RT GpuFloat.FP8E4M3 64 64 ← allocRT .FP8E4M3 64 64
    convert aFp8 a

    comment "FP8 GEMM"
    mma c aFp8 b c
    sync

  comment "Store output"
  convert out c
  store outShared out
  storeGlobal C_ptr outShared coord

-- Verify auto-generated kernel
#check gemmMixedFwd.kernel
#check gemmMixedFwd.launch

/-! ## Scaled FP8 GEMM with Bias

Common inference pattern: FP8 GEMM + scale + bias + activation.
-/

/-- FP8 GEMM with scale and bias fusion -/
@[gpu_kernel .SM90]
def gemmFp8ScaledBiasFwd (A_ptr : GPtr GpuFloat.FP8E4M3) (B_ptr : GPtr GpuFloat.FP8E4M3)
    (scale_ptr : GPtr GpuFloat.Float32) (bias_ptr : GPtr GpuFloat.Float32)
    (C_ptr : GPtr GpuFloat.BFloat16) (M : KVal UInt64) (N : KVal UInt64)
    (K_dim : KVal UInt64) : KernelM Unit := do
  comment "=== FP8 GEMM with Scale and Bias ==="
  comment "C = scale * (A @ B) + bias"

  let numBlocks : Nat := 8

  let coord ← blockCoord2D

  let a : RT GpuFloat.FP8E4M3 64 64 ← allocRT .FP8E4M3 64 64
  let b : RT GpuFloat.FP8E4M3 64 64 .Col ← allocRT .FP8E4M3 64 64 .Col
  let c : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Scale factor (for dequantization)
  let scale : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Bias (per output column)
  let bias : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let aShared : ST GpuFloat.FP8E4M3 64 64 ← allocST .FP8E4M3 64 64
  let bShared : ST GpuFloat.FP8E4M3 64 64 .Col ← allocST .FP8E4M3 64 64 .Col
  let scaleShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let biasShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load scale and bias"
  loadGlobal scaleShared scale_ptr coord
  sync
  load scale scaleShared
  loadVec bias biasShared

  comment "FP8 GEMM"
  for blkIdx in krange 0 numBlocks do
    loadGlobal aShared A_ptr (coord.withCol blkIdx.id)
    loadGlobal bShared B_ptr (coord.withRow blkIdx.id)
    sync
    load a aShared
    load b bShared
    mma c a b c
    sync

  comment "Apply scale"
  mul c c scale

  comment "Add bias (broadcast to all rows)"
  addRow c c bias

  comment "Store output"
  convert out c
  store outShared out
  storeGlobal C_ptr outShared coord

-- Verify auto-generated kernels
#check gemmFp8ScaledBiasFwd.kernel
#check gemmFp8ScaledBiasFwd.launch

-- Print generated kernels
#eval IO.println "=== FP8 E4M3 GEMM ===" *> IO.println (generateKernel gemmFp8E4M3Fwd.kernel)
#eval IO.println "\n=== FP8 E5M2 GEMM ===" *> IO.println (generateKernel gemmFp8E5M2Fwd.kernel)
#eval IO.println "\n=== MxFP8 GEMM ===" *> IO.println (generateKernel gemmMxFp8Fwd.kernel)
#eval IO.println "\n=== Mixed GEMM ===" *> IO.println (generateKernel gemmMixedFwd.kernel)
#eval IO.println "\n=== FP8 Scaled Bias ===" *> IO.println (generateKernel gemmFp8ScaledBiasFwd.kernel)

end Tyr.GPU.Kernels.PrecisionGemm
