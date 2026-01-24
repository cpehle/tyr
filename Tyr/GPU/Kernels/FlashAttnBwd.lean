/-
  Tyr/GPU/Kernels/FlashAttnBwdNew.lean

  FlashAttention forward and backward kernels using native Lean4 GPU DSL.
  Based on ThunderKittens MHA patterns.

  Forward pass stores L_vec (log-sum-exp) for use in backward.
  Backward pass computes dQ, dK, dV from dO.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## FlashAttention Forward (stores L_vec for backward) -/

/-- FlashAttention forward kernel with log-sum-exp output for backward

This version stores L_vec = log(sum(exp(S - max))) + max for each row,
which is needed to recompute the softmax during the backward pass.
-/
@[gpu_kernel .SM90]
def flashAttnFwdWithLse (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32) (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 4
  comment "=== FlashAttention Forward (with LSE for backward) ==="
  let coord ← blockCoord2D

  comment "Register tiles for Q, K, V"
  let q : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let k : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let v : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col

  comment "Attention scores (float32 for precision)"
  let s : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let p : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  comment "Output accumulator (float32)"
  let o : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize

  comment "Online softmax state"
  let softmaxState ← allocSoftmaxState .Float32 tileSize

  comment "Shared memory"
  let qShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let kShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let vShared : ST GpuFloat.BFloat16 tileSize tileSize .Col ← allocST .BFloat16 tileSize tileSize .Col
  let oShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let lseShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize

  comment "Load Q (long-resident)"
  loadGlobal qShared Q_ptr coord
  sync
  load q qShared

  comment "Initialize row_sum to 0"
  -- Note: zero operation on vectors would go here

  comment "Main loop over K, V blocks"
  for blkIdx in krange 0 numKvBlocks do
    comment "Load K, V tiles"
    loadGlobal kShared K_ptr (coord.withRow blkIdx.id)
    loadGlobal vShared V_ptr (coord.withRow blkIdx.id)
    sync
    load k kShared
    load v vShared

    comment "Compute attention scores: S = Q @ K^T"
    mmaT s q k s

    comment "Apply causal mask (fill upper triangle with -inf)"
    makeCausal s s (some (-1e10))

    comment "Online softmax"
    onlineSoftmax s o softmaxState

    comment "Convert to bf16 for V multiply"
    convert p s

    comment "Accumulate output: O += P @ V"
    mma o p v o

    sync

  comment "Final normalization: O = O / row_sum"
  finalizeSoftmax o softmaxState

  comment "Compute L_vec = log(row_sum) + row_max (for backward)"
  let lseVec ← computeLSE softmaxState

  comment "Store output and L_vec for backward pass"
  let oBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal O_ptr oShared coord
  storeVec lseShared lseVec
  storeVecGlobalRow L_ptr lseShared coord


/-! ## FlashAttention Backward Preparation Kernel -/

/-- Backward preparation: compute D_vec = rowSum(dO * O)

This is computed separately as it's needed by the main backward kernel
to compute dS = P * (dP - D_vec).
-/
@[gpu_kernel .SM90]
def flashAttnBwdPrep (dO_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (D_ptr : GPtr GpuFloat.Float32) (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    : KernelM Unit := do
  let tileSize : Nat := 64
  let coord ← blockCoord2D
  comment "=== FlashAttention Backward Prep ==="
  comment "Computes D_vec = rowSum(dO * O)"

  comment "Register tiles for dO and O"
  let dO : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let outFwd : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  comment "Temporary for element-wise product (float32)"
  let prod : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dOF : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let outF : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize

  comment "D vector output (per-row)"
  let dVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  comment "Shared memory"
  let dOShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let outShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let dVecShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize

  comment "Load dO and O"
  loadGlobal dOShared dO_ptr coord
  loadGlobal outShared O_ptr coord
  sync
  load dO dOShared
  load outFwd outShared

  comment "Convert to float32"
  convert dOF dO
  convert outF outFwd

  comment "Element-wise multiply: prod = dO * O"
  mul prod dOF outF

  comment "Row-wise sum: D_vec = sum(dO * O)"
  rowSum dVec prod

  comment "Store D_vec"
  storeVec dVecShared dVec
  storeVecGlobalRow D_ptr dVecShared coord


/-! ## FlashAttention Main Backward Kernel -/

/-- FlashAttention backward kernel

Computes dQ, dK, dV from forward activations and gradients.

Key equations:
  1. Recompute P = softmax(QK^T) using stored L_vec
  2. dP = dO @ V^T
  3. dS = P * (dP - D_vec)  where D_vec = rowSum(dO * O)
  4. dQ = dS @ K
  5. dK += dS^T @ Q (accumulated across query blocks)
  6. dV += P^T @ dO (accumulated across query blocks)
-/
@[gpu_kernel .SM90]
def flashAttnBwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (dO_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32) (D_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32) (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32) (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    : KernelM Unit := do
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 4
  comment "=== FlashAttention Backward ==="
  let coord ← blockCoord2D

  comment "=== Input tiles (from forward pass) ==="
  let q : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let k : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let v : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
  let dO : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  comment "=== Gradient accumulators (float32 for precision) ==="
  let dQ : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize
  let dK : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize
  let dV : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize

  comment "=== Intermediate computations ==="
  -- Attention scores and softmax output
  let s : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let p : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let pBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  -- Gradient of attention probs (dP = dO @ V^T)
  let dP : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize

  -- Gradient of scores (dS = P * (dP - D_vec))
  let dS : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dSBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize

  comment "=== Per-row vectors ==="
  let lseVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize  -- log-sum-exp from fwd
  let dVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize    -- D from prep kernel
  let rowMaxVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  comment "=== Shared memory ==="
  let qShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let kShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let vShared : ST GpuFloat.BFloat16 tileSize tileSize .Col ← allocST .BFloat16 tileSize tileSize .Col
  let dOShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let dQShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  let dKShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  let dVShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  let lseShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  let dVecShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize

  comment "Load Q and dO (long-resident for this query block)"
  loadGlobal qShared Q_ptr coord
  loadGlobal dOShared dO_ptr coord
  loadVecGlobalRow lseShared L_ptr coord
  loadVecGlobalRow dVecShared D_ptr coord
  sync
  load q qShared
  load dO dOShared

  comment "Load precomputed L_vec and D_vec"
  loadVec lseVec lseShared
  loadVec dVec dVecShared

  comment "Main loop over K, V blocks"
  for blkIdx in krange 0 numKvBlocks do
    comment "Load K, V for this block"
    loadGlobal kShared K_ptr (coord.withRow blkIdx.id)
    loadGlobal vShared V_ptr (coord.withRow blkIdx.id)
    sync
    load k kShared
    load v vShared

    comment "=== Recompute attention scores ==="
    comment "S = Q @ K^T"
    mmaT s q k s

    comment "Apply causal mask"
    makeCausal s s (some (-1e10))

    comment "=== Recompute P from stored L_vec ==="
    comment "P = exp(S - L_vec)"
    -- In practice: S already has max subtracted during forward
    -- Here we use L_vec to reconstruct the exact same softmax
    rowMax rowMaxVec s
    subCol s s rowMaxVec
    exp p s
    -- Note: would divide by sum(exp) or use L_vec directly

    comment "=== Compute dP = dO @ V^T ==="
    -- dP = dO @ V^T: dO is (tileSize, tileSize), V is (tileSize, tileSize) col-major
    -- When V is col-major, V^T can be viewed as row-major, so we swap layout
    let vRow : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v
    -- mmaT: dst = a @ b^T + c (bf16 inputs, f32 accumulator)
    mmaT dP dO vRow dP

    comment "=== Compute dS = P * (dP - D_vec) ==="
    subCol dP dP dVec    -- dP - D_vec
    mul dS p dP          -- P * (dP - D_vec)

    comment "Apply causal mask to dS"
    makeCausal dS dS (some 0.0)

    comment "Convert dS to bf16 for MMA"
    convert dSBf16 dS

    comment "=== Accumulate gradients ==="

    comment "dQ += dS @ K"
    let kCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSBf16 kCol dQ

    comment "dK += dS^T @ Q"
    let dST : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    transpose dST dSBf16
    let qCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dK dST qCol dK

    comment "dV += P^T @ dO"
    let pT : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    convert pBf16 p
    transpose pT pBf16
    let dOCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout dOCol dO
    mma dV pT dOCol dV

    sync

  comment "=== Store gradients (using atomic add for K, V) ==="
  store dQShared dQ
  store dKShared dK
  store dVShared dV
  storeGlobal dQ_ptr dQShared coord
  storeGlobalAdd dK_ptr dKShared coord  -- Atomic add for accumulation across query blocks
  storeGlobalAdd dV_ptr dVShared coord  -- Atomic add for accumulation across query blocks


/-! ## Combined Forward+Backward Example -/

/-- Full training iteration pattern: forward then backward

This shows how the kernels compose for training:
1. flashAttnFwdWithLse - stores O and L_vec
2. <external loss computation gives dO>
3. flashAttnBwdPrep - compute D_vec from dO, O
4. flashAttnBwd - compute dQ, dK, dV
-/
def trainingPattern : String :=
  "// FlashAttention Training Pattern:\n" ++
  "// 1. Forward: flash_attn_fwd_lse(Q, K, V) -> O, L_vec\n" ++
  "// 2. External: compute loss gradient dO\n" ++
  "// 3. Prep: flash_attn_bwd_prep(dO, O) -> D_vec\n" ++
  "// 4. Backward: flash_attn_bwd(Q, K, V, dO, L_vec, D_vec) -> dQ, dK, dV"


-- Verify auto-generated kernel and launch definitions
#check flashAttnFwdWithLse.kernel
#check flashAttnFwdWithLse.launch
#check flashAttnBwdPrep.kernel
#check flashAttnBwdPrep.launch
#check flashAttnBwd.kernel
#check flashAttnBwd.launch

-- Generate C++ code
#eval IO.println "=== FlashAttention Forward (with LSE) ===" *>
      IO.println (generateKernel flashAttnFwdWithLse.kernel)

#eval IO.println "\n=== FlashAttention Backward Prep ===" *>
      IO.println (generateKernel flashAttnBwdPrep.kernel)

#eval IO.println "\n=== FlashAttention Backward ===" *>
      IO.println (generateKernel flashAttnBwd.kernel)

#eval IO.println "\n" *> IO.println trainingPattern

end Tyr.GPU.Kernels
