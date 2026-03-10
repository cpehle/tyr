/-
  Tyr/GPU/Kernels/Flux.lean

  Flux fused matmul + activation kernels aligned to the vendored
  `flux_gelu.cu` and `flux_gate.cu` kernels.

  Canonical source-facing surfaces:

  - `Tyr.GPU.Kernels.tkFluxMatmulGeluFwd`
  - `Tyr.GPU.Kernels.tkFluxMatmulGateFwd`

  The older `fluxGeluFwd`, `fluxGateFwd`, and `fluxSiluFwd` entries remain as
  convenience kernels, but they are not the best approximation of the vendored
  ThunderKittens signatures.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.Flux

open Tyr.GPU
open Tyr.GPU.Codegen

private abbrev fluxTileRows : Nat := 64
private abbrev fluxTileCols : Nat := 64
private abbrev fluxKBlocks : Nat := 8

private def loadColumnVecF32
    (ptr : GPtr GpuFloat.BFloat16) (coord : RTileCoord)
    : KernelM (RV GpuFloat.Float32 fluxTileCols) := do
  let shared : SV GpuFloat.BFloat16 fluxTileCols ← allocSV .BFloat16 fluxTileCols
  let vecBf : RV GpuFloat.BFloat16 fluxTileCols ← allocRV .BFloat16 fluxTileCols
  let vecF : RV GpuFloat.Float32 fluxTileCols ← allocRV .Float32 fluxTileCols
  loadVecGlobalCol shared ptr coord
  loadVec vecBf shared
  convertVec vecF vecBf
  pure vecF

private def fluxMatmulAccumulator
    (lhs_ptr : GPtr GpuFloat.BFloat16)
    (rhs_ptr : GPtr GpuFloat.BFloat16)
    (coord : RTileCoord)
    : KernelM (RT GpuFloat.Float32 fluxTileRows fluxTileCols) := do
  let x : RT GpuFloat.BFloat16 fluxTileRows fluxTileCols ← allocRT .BFloat16 fluxTileRows fluxTileCols
  let w : RT GpuFloat.BFloat16 fluxTileRows fluxTileCols .Col ← allocRT .BFloat16 fluxTileRows fluxTileCols .Col
  let acc : RT GpuFloat.Float32 fluxTileRows fluxTileCols ← zeroRT .Float32 fluxTileRows fluxTileCols
  let xShared : ST GpuFloat.BFloat16 fluxTileRows fluxTileCols ← allocST .BFloat16 fluxTileRows fluxTileCols
  let wShared : ST GpuFloat.BFloat16 fluxTileRows fluxTileCols .Col ← allocST .BFloat16 fluxTileRows fluxTileCols .Col

  loadGlobal xShared lhs_ptr coord
  sync
  for kIdx in krange 0 fluxKBlocks do
    loadGlobal wShared rhs_ptr (coord.withCol kIdx.id)
    sync
    load x xShared
    load w wShared
    mma acc x w acc
    sync
  pure acc

/-! ## Canonical ThunderKittens Surfaces -/

/-- Canonical `flux_gelu.cu`-shaped fused matmul + GELU surface.

This surface adds the missing per-column bias vector that the vendored CUDA
kernel carries through the prototype layout before applying the GELU epilogue. -/
@[gpu_kernel .SM90]
def tkFluxMatmulGeluFwd
    (lhs_ptr : GPtr GpuFloat.BFloat16)
    (rhs_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (acc_ptr : GPtr GpuFloat.BFloat16)
    (_m : KVal UInt64) (_n : KVal UInt64) (_k : KVal UInt64) : KernelM Unit := do
  comment "=== ThunderKittens flux_gelu.cu: fused matmul + bias + GELU ==="

  let coord ← blockCoord2D
  let acc ← fluxMatmulAccumulator lhs_ptr rhs_ptr coord
  let bias ← loadColumnVecF32 bias_ptr coord
  let geluIn : RT GpuFloat.Float32 fluxTileRows fluxTileCols ← allocRT .Float32 fluxTileRows fluxTileCols
  let sq : RT GpuFloat.Float32 fluxTileRows fluxTileCols ← allocRT .Float32 fluxTileRows fluxTileCols
  let tanhArg : RT GpuFloat.Float32 fluxTileRows fluxTileCols ← allocRT .Float32 fluxTileRows fluxTileCols
  let out : RT GpuFloat.BFloat16 fluxTileRows fluxTileCols ← allocRT .BFloat16 fluxTileRows fluxTileCols
  let outShared : ST GpuFloat.BFloat16 fluxTileRows fluxTileCols ← allocST .BFloat16 fluxTileRows fluxTileCols

  addRow geluIn acc bias
  mul sq geluIn geluIn
  scalarMul sq sq 0.044715
  scalarAdd sq sq 1.0
  scalarMul tanhArg geluIn 0.79788456
  mul tanhArg tanhArg sq
  fastTanh tanhArg tanhArg
  scalarAdd tanhArg tanhArg 1.0
  scalarMul geluIn geluIn 0.5
  mul geluIn geluIn tanhArg

  convert out geluIn
  store outShared out
  sync
  storeGlobal acc_ptr outShared coord

/-- Canonical `flux_gate.cu`-shaped fused matmul + bias + gate + residual
surface.

The vendored kernel multiplies the bias-adjusted accumulator by a per-column
gate vector and then adds a pre-existing output tile `y`. This Lean surface
keeps that contract explicit. -/
@[gpu_kernel .SM90]
def tkFluxMatmulGateFwd
    (lhs_ptr : GPtr GpuFloat.BFloat16)
    (rhs_ptr : GPtr GpuFloat.BFloat16)
    (bias_ptr : GPtr GpuFloat.BFloat16)
    (gate_ptr : GPtr GpuFloat.BFloat16)
    (y_ptr : GPtr GpuFloat.BFloat16)
    (acc_ptr : GPtr GpuFloat.BFloat16)
    (_m : KVal UInt64) (_n : KVal UInt64) (_k : KVal UInt64) : KernelM Unit := do
  comment "=== ThunderKittens flux_gate.cu: fused matmul + bias + gate + residual ==="

  let coord ← blockCoord2D
  let acc ← fluxMatmulAccumulator lhs_ptr rhs_ptr coord
  let bias ← loadColumnVecF32 bias_ptr coord
  let gate ← loadColumnVecF32 gate_ptr coord
  let gated : RT GpuFloat.Float32 fluxTileRows fluxTileCols ← allocRT .Float32 fluxTileRows fluxTileCols
  let yBf : RT GpuFloat.BFloat16 fluxTileRows fluxTileCols ← allocRT .BFloat16 fluxTileRows fluxTileCols
  let yF : RT GpuFloat.Float32 fluxTileRows fluxTileCols ← allocRT .Float32 fluxTileRows fluxTileCols
  let out : RT GpuFloat.BFloat16 fluxTileRows fluxTileCols ← allocRT .BFloat16 fluxTileRows fluxTileCols
  let yShared : ST GpuFloat.BFloat16 fluxTileRows fluxTileCols ← allocST .BFloat16 fluxTileRows fluxTileCols
  let outShared : ST GpuFloat.BFloat16 fluxTileRows fluxTileCols ← allocST .BFloat16 fluxTileRows fluxTileCols

  addRow gated acc bias
  mulRow gated gated gate
  loadGlobal yShared y_ptr coord
  sync
  load yBf yShared
  convert yF yBf
  add gated gated yF

  convert out gated
  store outShared out
  sync
  storeGlobal acc_ptr outShared coord

/-! ## Flux GELU Kernel

Fused linear + GELU activation using producer-consumer pattern.
GELU formula: f * 0.5 * (1 + fast_tanh(f * 0.79788456 * (1 + f² * 0.044715)))
-/

/-- Flux GELU forward pass - fused linear + GELU

Parameters:
  X_ptr: Input tensor [batch, seq_len, hidden_dim]
  W_ptr: Weight matrix [hidden_dim, out_dim]
  O_ptr: Output tensor [batch, seq_len, out_dim]
  batch_size: Number of sequences in batch
  seq_len: Sequence length
  hidden_dim: Input hidden dimension (must be multiple of 64)
  out_dim: Output dimension (must be multiple of 64)
-/
@[gpu_kernel .SM90]
def fluxGeluFwd (X_ptr : GPtr GpuFloat.BFloat16) (W_ptr : GPtr GpuFloat.BFloat16)
    (O_ptr : GPtr GpuFloat.BFloat16)
    (_batch_size : KVal UInt64) (_seq_len : KVal UInt64)
    (_hidden_dim : KVal UInt64) (_out_dim : KVal UInt64) : KernelM Unit := do
  comment "=== Flux GELU Forward (Fused Linear + GELU) ==="

  -- Get block coordinates: (batch_head_idx, seq_block_idx)
  let coord ← blockCoord2D

  -- Input tile (64 tokens × 64 hidden)
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Weight tile (col-major for tensor cores)
  let w : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Accumulator (float32 for precision)
  let acc : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Working tiles for GELU computation
  let f : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let f2 : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let tanh_arg : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let wShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load input tile from global memory"
  loadGlobal xShared X_ptr coord
  sync

  comment "Accumulate over hidden dimension tiles (hidden_dim / 64 iterations)"
  for kIdx in krange 0 8 do
    comment "Load weight tile for this k-block"
    loadGlobal wShared W_ptr (coord.withCol kIdx.id)
    sync
    load x xShared
    load w wShared

    comment "Linear: acc += x @ w"
    mma acc x w acc
    sync

  comment "Convert to float32 for GELU computation"
  convert f acc

  comment "GELU: f * 0.5 * (1 + fast_tanh(f * 0.79788456 * (1 + f² * 0.044715)))"
  -- f² = f * f
  mul f2 f f
  -- f² * 0.044715
  scalarMul f2 f2 0.044715
  -- 1 + f² * 0.044715
  scalarAdd f2 f2 1.0
  -- f * 0.79788456 * (1 + f² * 0.044715)
  scalarMul tanh_arg f 0.79788456
  mul tanh_arg tanh_arg f2
  -- fast_tanh(...)
  fastTanh tanh_arg tanh_arg
  -- 1 + fast_tanh(...)
  scalarAdd tanh_arg tanh_arg 1.0
  -- f * 0.5 * (1 + fast_tanh(...))
  scalarMul f f 0.5
  mul f f tanh_arg

  comment "Convert back to bf16 and store to global memory"
  convert out f
  store outShared out
  sync
  storeGlobal O_ptr outShared coord

-- Verify auto-generated kernel

/-! ## Flux Gate Kernel (SwiGLU-style)

Gate multiplication with residual addition.
-/

/-- Flux Gate forward pass - gating mechanism

Parameters:
  X_ptr: Main activation tensor [batch, seq_len, hidden_dim]
  Gate_ptr: Gate tensor [batch, seq_len, hidden_dim]
  Residual_ptr: Residual input tensor [batch, seq_len, hidden_dim]
  O_ptr: Output tensor [batch, seq_len, hidden_dim]
  batch_size: Number of sequences in batch
  seq_len: Sequence length
  hidden_dim: Hidden dimension (must be multiple of 64)
-/
@[gpu_kernel .SM90]
def fluxGateFwd (X_ptr : GPtr GpuFloat.BFloat16) (Gate_ptr : GPtr GpuFloat.BFloat16)
    (Residual_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (_batch_size : KVal UInt64) (_seq_len : KVal UInt64)
    (_hidden_dim : KVal UInt64) : KernelM Unit := do
  comment "=== Flux Gate Forward (SwiGLU-style) ==="

  -- Get block coordinates: (batch_idx, seq_block_idx)
  let coord ← blockCoord2D

  -- Main activation tile
  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Gate tile
  let gate : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Residual input
  let residual : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Working tiles (float32 for precision)
  let acc : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let gateF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let residualF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Output
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let gateShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let residualShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Process hidden dimension tiles"
  for hiddenIdx in krange 0 16 do
    let tileCoord := coord.withCol hiddenIdx.id

    comment "Load inputs from global memory"
    loadGlobal xShared X_ptr tileCoord
    loadGlobal gateShared Gate_ptr tileCoord
    loadGlobal residualShared Residual_ptr tileCoord
    sync

    load x xShared
    load gate gateShared
    load residual residualShared

    comment "Convert to float32"
    convert acc x
    convert gateF gate
    convert residualF residual

    comment "Gate multiplication: acc = acc * gate"
    mul acc acc gateF

    comment "Residual addition: acc = acc + residual"
    add acc acc residualF

    comment "Convert back and store to global memory"
    convert out acc
    store outShared out
    sync
    storeGlobal O_ptr outShared tileCoord

/-! ## Flux Linear + SiLU (GLU variant)

Linear transformation followed by SiLU gating.
-/

/-- Flux Linear + SiLU forward pass (SwiGLU MLP)

Parameters:
  X_ptr: Input tensor [batch, seq_len, hidden_dim]
  W_up_ptr: Up-projection weight [hidden_dim, intermediate_dim]
  W_gate_ptr: Gate-projection weight [hidden_dim, intermediate_dim]
  O_ptr: Output tensor [batch, seq_len, intermediate_dim]
  batch_size: Number of sequences in batch
  seq_len: Sequence length
  hidden_dim: Input hidden dimension (must be multiple of 64)
  intermediate_dim: Intermediate/output dimension (must be multiple of 64)
-/
@[gpu_kernel .SM90]
def fluxSiluFwd (X_ptr : GPtr GpuFloat.BFloat16) (W_up_ptr : GPtr GpuFloat.BFloat16)
    (W_gate_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (_batch_size : KVal UInt64) (_seq_len : KVal UInt64)
    (_hidden_dim : KVal UInt64) (_intermediate_dim : KVal UInt64) : KernelM Unit := do
  comment "=== Flux Linear + SiLU (SwiGLU MLP) ==="

  -- Get block coordinates: (batch_seq_idx, out_tile_idx)
  let coord ← blockCoord2D

  let x : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let wUp : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let wGate : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  let up : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let gateVal : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let xShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let wUpShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let wGateShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load input tile"
  loadGlobal xShared X_ptr coord
  sync

  comment "Accumulate over hidden dimension tiles (hidden_dim / 64 iterations)"
  for kIdx in krange 0 8 do
    let weightCoord := coord.withCol kIdx.id

    comment "Load weight tiles for this k-block"
    loadGlobal wUpShared W_up_ptr weightCoord
    loadGlobal wGateShared W_gate_ptr weightCoord
    sync

    load x xShared
    load wUp wUpShared
    load wGate wGateShared

    comment "Compute up projection: up += x @ wUp"
    mma up x wUp up

    comment "Compute gate projection: gate += x @ wGate"
    mma gateVal x wGate gateVal
    sync

  comment "Apply SiLU to gate: gate = gate * sigmoid(gate)"
  silu gateVal gateVal

  comment "Multiply: out = up * silu(gate)"
  mul up up gateVal

  comment "Store output to global memory"
  convert out up
  store outShared out
  sync
  storeGlobal O_ptr outShared coord

-- Verify auto-generated kernel

-- Print generated kernels

end Tyr.GPU.Kernels.Flux

namespace Tyr.GPU.Kernels

/-- Canonical ThunderKittens-aligned flux GELU surface. -/
abbrev tkFluxMatmulGeluFwd := Flux.tkFluxMatmulGeluFwd

/-- Canonical ThunderKittens-aligned flux gate surface. -/
abbrev tkFluxMatmulGateFwd := Flux.tkFluxMatmulGateFwd

end Tyr.GPU.Kernels
