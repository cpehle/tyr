/-
  Tyr/GPU/Kernels/LayerNormNew.lean

  LayerNorm kernel using native Lean4 GPU DSL.
  Type-safe with compile-time dimension checking.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-- LayerNorm forward using tile-based computation

Uses 64x64 tiles where each row is a token and columns span hidden dim.
This is more efficient than vector-based for larger hidden dimensions.
-/
def layerNormTiledNew (tileSize : Nat := 64) : KernelM Unit := do
  comment "=== LayerNorm Forward (Tiled) ==="

  comment "Register tiles for input/output"
  let x : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let xf : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let temp : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize

  comment "Statistics vectors (one per row/token)"
  let mean : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let var : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let invStd : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  comment "Normalization parameters"
  let weight : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let bias : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  comment "Shared memory"
  let xShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let weightShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let biasShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize

  comment "Load normalization parameters (long-resident)"
  load weight weightShared
  load bias biasShared

  comment "Process tiles of tokens"
  forLoop 0 16 do
    comment "Load input tile"
    load x xShared

    comment "Convert to float32 for precision"
    convert xf x

    comment "Step 1: Compute row-wise mean"
    rowSum mean xf
    -- Note: Would need scalar division by tileSize

    comment "Step 2: Subtract mean from each row"
    subCol temp xf mean

    comment "Step 3: Compute variance = mean((x - mean)^2)"
    mul xf temp temp
    rowSum var xf
    -- Note: Would need scalar division + eps

    comment "Step 4: Compute inverse std = 1/sqrt(var + eps)"
    -- rsqrt invStd var  -- Would need vec rsqrt

    comment "Step 5: Normalize: (x - mean) * invStd"
    mulCol temp temp invStd

    comment "Step 6: Scale and shift"
    let weightF : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
    let biasF : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
    convert weightF weight
    convert biasF bias
    mul temp temp weightF
    add temp temp biasF

    comment "Convert back to bf16 and store"
    convert x temp
    store xShared x

    sync

/-- Build the LayerNorm kernel -/
def layerNormTiledKernel : Kernel :=
  buildKernelM "layernorm_tiled_fwd" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "weight_ptr", dtype := .BFloat16, isPointer := true },
    { name := "bias_ptr", dtype := .BFloat16, isPointer := true },
    { name := "out_ptr", dtype := .BFloat16, isPointer := true },
    { name := "batch_size", dtype := .Float32, isPointer := false },
    { name := "hidden_dim", dtype := .Float32, isPointer := false }
  ] (layerNormTiledNew 64)

/-- RMSNorm forward (simpler variant without mean subtraction) -/
def rmsNormTiledNew (tileSize : Nat := 64) : KernelM Unit := do
  comment "=== RMSNorm Forward (Tiled) ==="

  let x : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let xf : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let temp : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize

  let rmsSq : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let invRms : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  let weight : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  let xShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let weightShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize

  comment "Load weight (long-resident)"
  load weight weightShared

  comment "Process tiles"
  forLoop 0 16 do
    comment "Load input"
    load x xShared

    comment "Convert to float32"
    convert xf x

    comment "Compute sum of squares per row"
    mul temp xf xf
    rowSum rmsSq temp

    comment "Compute inverse RMS (would need vec rsqrt)"
    -- rsqrtVec invRms rmsSq

    comment "Normalize: x * invRms"
    mulCol temp xf invRms

    comment "Scale by weight"
    let weightF : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
    convert weightF weight
    mul temp temp weightF

    comment "Convert back and store"
    convert x temp
    store xShared x

    sync

/-- Build the RMSNorm kernel -/
def rmsNormTiledKernel : Kernel :=
  buildKernelM "rmsnorm_tiled_fwd" .SM90 #[
    { name := "x_ptr", dtype := .BFloat16, isPointer := true },
    { name := "weight_ptr", dtype := .BFloat16, isPointer := true },
    { name := "out_ptr", dtype := .BFloat16, isPointer := true },
    { name := "hidden_dim", dtype := .Float32, isPointer := false }
  ] (rmsNormTiledNew 64)

-- Generate C++ code
#eval IO.println "=== LayerNorm Tiled ===" *> IO.println (generateKernel layerNormTiledKernel)
#eval IO.println "\n=== RMSNorm Tiled ===" *> IO.println (generateKernel rmsNormTiledKernel)

end Tyr.GPU.Kernels
