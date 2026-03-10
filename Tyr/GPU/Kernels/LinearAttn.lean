/-
  Tyr/GPU/Kernels/LinearAttn.lean

  ThunderKittens-shaped decayed linear attention kernels.

  The vendored reference in
  `thirdparty/ThunderKittens/kernels/linear_attention/linear_attention.cu`
  is not a generic feature-map attention kernel. It has two paths per chunk:

  - a local causal `QK^T V` path with exponential row/column decay, and
  - a recurrent `KV` state path decayed across blocks.

  The canonical kernel below matches that structure. To keep the Lean DSL
  surface explicit, it takes precomputed decay vectors instead of constructing
  them from `slope` in-kernel.
-/

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.LinearAttn

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Load one 64-token chunk of Q/K/V tiles. -/
private def loadQKVChunk
    (qShared : ST GpuFloat.BFloat16 64 64 .Row)
    (kShared : ST GpuFloat.BFloat16 64 64 .Row)
    (vShared : ST GpuFloat.BFloat16 64 64 .Col)
    (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16)
    (coord : RTileCoord) : KernelM Unit := do
  loadGlobal qShared Q_ptr coord
  loadGlobal kShared K_ptr coord
  loadGlobal vShared V_ptr coord

/-- Convert a Float32 row tile to a BF16 col-layout tile for `mma`. -/
private def toBf16Col
    (src : RT GpuFloat.Float32 64 64 .Row) : KernelM (RT GpuFloat.BFloat16 64 64 .Col) := do
  let bfRow : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let bfCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  convert bfRow src
  swapLayout bfCol bfRow
  pure bfCol

/-- Update the recurrent `K^T V` state after applying block decay. -/
private def updateStateKtV
    (state : RT GpuFloat.Float32 64 64 .Row)
    (kF : RT GpuFloat.Float32 64 64 .Row)
    (v : RT GpuFloat.BFloat16 64 64 .Col)
    (kDecay : RV GpuFloat.Float32 64)
    (stateDecay : RV GpuFloat.Float32 64) : KernelM Unit := do
  let kDecayed : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let kDecayedBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kDecayedCol : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let updated : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  mulCol state state stateDecay
  mulCol kDecayed kF kDecay
  convert kDecayedBf kDecayed
  swapLayout kDecayedCol kDecayedBf
  mmaAtB updated kDecayedCol v state
  copy state updated

/-- Decayed causal linear attention forward.

`q_decay_ptr`, `k_decay_ptr`, and `state_decay_ptr` are explicit precomputed
decay vectors for the current head/chunk family:

- `q_decay_ptr`: row decay applied to local scores and recurrent queries
- `k_decay_ptr`: row decay applied to keys before the `K^T V` state update
- `state_decay_ptr`: repeated block decay applied to the recurrent state
-/
@[gpu_kernel .SM90]
def linearAttnFwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (state_ptr : GPtr GpuFloat.Float32)
    (q_decay_ptr : GPtr GpuFloat.Float32) (k_decay_ptr : GPtr GpuFloat.Float32)
    (state_decay_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64) (_head_dim : KVal UInt64) : KernelM Unit := do
  comment "=== Decayed Linear Attention Forward ==="
  comment "Local causal decayed QK^T V + recurrent decayed KV state"

  let coord ← blockCoord2D
  let numChunks : Nat := 16

  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  let qF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let kF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let qDecayed : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let qDecayedBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let state : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let stateBf : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  let localScores : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let localScoresBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let o : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let outBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let qDecay : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let kDecay : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let stateDecay : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let stateShared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let qDecayShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let kDecayShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let stateDecayShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load recurrent state and decay vectors"
  loadGlobal stateShared state_ptr coord
  loadVecGlobalCoord qDecayShared q_decay_ptr coord.c
  loadVecGlobalCoord kDecayShared k_decay_ptr coord.c
  loadVecGlobalCoord stateDecayShared state_decay_ptr coord.c
  sync
  load state stateShared
  loadVec qDecay qDecayShared
  loadVec kDecay kDecayShared
  loadVec stateDecay stateDecayShared

  for chunkIdx in krange 0 numChunks do
    let chunkCoord := coord.withRow chunkIdx.id
    loadQKVChunk qShared kShared vShared Q_ptr K_ptr V_ptr chunkCoord
    sync
    load q qShared
    load k kShared
    load v vShared
    convert qF q
    convert kF k
    zero o

    comment "Local decayed causal attention path"
    zero localScores
    mmaT localScores q k localScores
    makeCausal localScores localScores (some 0.0)
    mulCol localScores localScores qDecay
    mulRow localScores localScores kDecay
    convert localScoresBf localScores
    mma o localScoresBf v o

    comment "Recurrent decayed state path"
    mulCol qDecayed qF qDecay
    convert qDecayedBf qDecayed
    let stateBf' ← toBf16Col state
    copy stateBf stateBf'
    let recurrent : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    mma recurrent qDecayedBf stateBf recurrent
    add o o recurrent

    comment "State update: state = decay(state) + K^T V"
    updateStateKtV state kF v kDecay stateDecay

    convert outBf o
    store outShared outBf
    storeGlobal O_ptr outShared chunkCoord
    sync

  store stateShared state
  storeGlobal state_ptr stateShared coord

/-- Canonical ThunderKittens-aligned name. -/
abbrev tkLinearAttnFwd := linearAttnFwd

end Tyr.GPU.Kernels.LinearAttn
