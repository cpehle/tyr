/-
  Tyr/GPU/Kernels/Flux.lean

  Flux activation kernels (fused linear + GELU/Gate).
  Based on ThunderKittens patterns.

  Key features:
  - Producer-consumer pattern with TMA loads
  - Fused linear layer + activation
  - Hardware-accelerated fast_tanh for GELU
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

namespace Tyr.GPU.Kernels.Flux

open Tyr.GPU
open Tyr.GPU.Codegen

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
#check fluxGeluFwd.kernel
#check fluxGeluFwd.launch

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
#check fluxSiluFwd.kernel
#check fluxSiluFwd.launch

-- Print generated kernels
#eval IO.println "=== Flux GELU ===" *> IO.println (generateKernel fluxGeluFwd.kernel)
#eval IO.println "\n=== Flux Gate ===" *> IO.println (generateKernel fluxGateFwd.kernel)
#eval IO.println "\n=== Flux SiLU ===" *> IO.println (generateKernel fluxSiluFwd.kernel)

end Tyr.GPU.Kernels.Flux
