/-
  Tyr/GPU/Kernels/Mamba2.lean

  Mamba2 state-space model forward kernels.
  Based on ThunderKittens patterns:
  - Hillis-Steele prefix sum for cumulative decay
  - Exponential state decay computation
  - Attention with decay masking
  - State accumulation across chunks

  Backward ownership lives in `Tyr.GPU.Kernels.MambaBwd`.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.Mamba2

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Mamba2 Forward Kernel

The Mamba2 architecture uses selective state spaces with:
1. Per-position decay factors (A vector)
2. Input-dependent state updates
3. Causal attention with exponential decay

Key computation flow:
1. Compute cumulative sum of decay factors (log-space)
2. Convert to decay matrix: decay[i,j] = exp(cumsum[i] - cumsum[j])
3. Apply causal mask to decay matrix
4. Compute attention with decay: O = softmax(Q @ K^T * decay) @ V
5. Update running state: KV_state = KV_state * total_decay + K^T @ V
-/

/-- Mamba2 forward pass - single chunk processing -/
@[gpu_kernel .SM90]
def mamba2Fwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (A_ptr : GPtr GpuFloat.Float32)
    (O_ptr : GPtr GpuFloat.BFloat16) (state_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64) : KernelM Unit := do
  let _ := (seq_len, head_dim)
  comment "=== Mamba2 Forward Pass ==="

  let numChunks : Nat := 8

  let coord ← blockCoord2D

  -- Register tiles for Q, K, V (64x64)
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let outBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Attention scores and decay
  let att : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let decay : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Decay vector (log-space cumulative sum)
  let aVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let cumsum : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let totalDecay : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let kDecay : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- State tiles (KV accumulator)
  let kv : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- State contribution temporaries
  let qF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let qScaled : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let qScaledBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let kScaled : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let kScaledBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let stateBfRow : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let stateBf : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let kCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let prefixRows : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let prefixCols : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let prefixHead : RT GpuFloat.Float32 1 64 ← allocRT .Float32 1 64
  let prefixLastCol : RT GpuFloat.Float32 64 1 ← allocRT .Float32 64 1

  -- Shared memory
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let aShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let stateShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load running KV state"
  loadGlobal stateShared state_ptr coord
  sync
  load kv stateShared

  comment "Main loop over sequence chunks"
  for chunkIdx in krange 0 numChunks do
    -- Load Q, K, V for this chunk
    loadGlobal qShared Q_ptr (coord.withRow chunkIdx.id)
    loadGlobal kShared K_ptr (coord.withRow chunkIdx.id)
    loadGlobal vShared V_ptr (coord.withRow chunkIdx.id)
    loadVecGlobalRow aShared A_ptr (coord.withRow chunkIdx.id)
    sync
    load q qShared
    load k kShared
    load v vShared
    loadVec aVec aShared

    comment "Step 1: Compute decay cumsum (Hillis-Steele scan)"
    broadcastRow prefixRows aVec
    cumsumRow prefixRows prefixRows
    sliceRows prefixHead prefixRows 0 1
    colSum cumsum prefixHead

    comment "Step 2: Compute decay matrix"
    broadcastRow prefixRows cumsum
    broadcastCol prefixCols cumsum
    sub decay prefixCols prefixRows
    exp decay decay

    comment "Step 3: Apply causal mask"
    makeCausal decay decay (some 0.0)

    comment "Step 4: Compute attention with decay"
    -- att = Q @ K^T
    mmaT att q k att
    -- Scale by decay
    mul att att decay

    comment "Step 5: Compute local output and recurrent state contribution"
    let attBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert attBf att
    zero o
    mma o attBf v o

    convert qF q
    expVec totalDecay cumsum
    mulCol qScaled qF totalDecay
    convert qScaledBf qScaled
    convert stateBfRow kv
    swapLayout stateBf stateBfRow
    mma o qScaledBf stateBf o

    comment "Step 6: Update state (KV accumulator)"
    sliceCols prefixLastCol prefixRows 63 1
    rowSum totalDecay prefixLastCol
    mulCol kv kv totalDecay

    subVec kDecay totalDecay cumsum
    expVec kDecay kDecay
    convert kF k
    mulCol kScaled kF kDecay
    convert kScaledBf kScaled
    swapLayout kCol kScaledBf
    mmaAtB kv kCol v kv

    comment "Store output"
    convert outBf o
    store outShared outBf
    storeGlobal O_ptr outShared (coord.withRow chunkIdx.id)

    sync

  comment "Store updated KV state"
  store stateShared kv
  storeGlobal state_ptr stateShared coord

-- Verify auto-generated kernel

end Tyr.GPU.Kernels.Mamba2
