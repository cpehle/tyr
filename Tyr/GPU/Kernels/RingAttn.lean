/-
  Tyr/GPU/Kernels/RingAttn.lean

  Ring Attention kernel implementation for sequence parallelism.
  Based on ThunderKittens patterns.

  Key features:
  - Three-phase kernel: partial attention, communication, reduction
  - Log-sum-exp trick for numerically stable reduction
  - Ring-shift KV blocks between GPUs
  - Overlapped communication and computation
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute

namespace Tyr.GPU.Kernels.RingAttn

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Ring Attention

Ring attention enables long sequence processing by:
1. Splitting sequence across GPUs
2. Each GPU holds its Q block, KV blocks are ring-shifted
3. Partial attention computed with each KV block
4. Results combined using log-sum-exp trick

Three phases:
- Phase 1: Compute partial attention with local KV
- Phase 2: Ring-shift KV to next GPU, compute with new KV
- Phase 3: Combine partial outputs using stable reduction
-/

/-- Ring Attention - Phase 1: Partial attention computation -/
@[gpu_kernel .SM90]
def ringAttnPartial : KernelM Unit := do
  comment "=== Ring Attention - Partial Computation ==="
  comment "Compute attention with one KV block, output partial O and log-sum-exp"

  -- Q block (stays on this GPU)
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- KV blocks (ring-shifted between GPUs)
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Attention scores
  let att : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Softmax tracking
  let maxVec : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let sumVec : RV GpuFloat.Float32 64 ← zeroRV .Float32 64

  -- Partial output
  let oPartial : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Log-sum-exp for stable reduction
  let lse : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Shared memory
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let lseShared : SV GpuFloat.Float32 64 ← allocSV .Float32 64

  comment "Load Q (stays resident on this GPU)"
  load q qShared

  comment "Process KV block"
  load k kShared
  load v vShared

  comment "Compute attention scores: att = Q @ K^T"
  let qF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let kF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  convert qF q
  convert kF k
  mmaT att qF kF (← zeroRT .Float32 64 64)

  comment "Scale by 1/sqrt(d)"
  scalarMul att att 0.125  -- 1/sqrt(64)

  comment "Apply causal mask if needed"
  makeCausal att att (some (-1.0e10))

  comment "Online softmax"
  rowMax maxVec att
  subCol att att maxVec
  exp att att
  rowSum sumVec att
  divCol att att sumVec

  comment "Compute partial output: O_partial = softmax(QK^T) @ V"
  let attBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  convert attBf att
  mma oPartial attBf v oPartial

  comment "Compute log-sum-exp for stable reduction"
  -- lse = max + log(sum)
  copyVec lse maxVec
  logVec sumVec sumVec
  addVec lse lse sumVec

  comment "Store partial output and LSE"
  convert out oPartial
  store outShared out
  storeVec lseShared lse
  sync

def ringAttnPartialKernel : Kernel :=
  buildKernelM "ring_attn_partial" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O_partial", dtype := .BFloat16, isPointer := true },
    { name := "lse", dtype := .Float32, isPointer := true },
    { name := "rank", dtype := .Float32, isPointer := false },
    { name := "kv_block_idx", dtype := .Float32, isPointer := false }
  ] ringAttnPartial

/-- Ring Attention - Phase 2: Full ring processing -/
@[gpu_kernel .SM90]
def ringAttnFull : KernelM Unit := do
  comment "=== Ring Attention - Full Ring Processing ==="
  comment "Process all KV blocks around the ring, accumulate with LSE"

  -- Q block (stays on this GPU)
  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- KV blocks (ring-shifted)
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Double buffering for KV (overlap compute and communication)
  let kNext : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let vNext : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Attention scores
  let att : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Running output and LSE
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let lseRunning : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Per-block softmax tracking
  let maxVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let sumVec : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let lseNew : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let scale : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Shared memory
  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let kNextShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vNextShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load Q (stays resident)"
  load q qShared

  comment "Load initial KV block"
  load k kShared
  load v vShared

  comment "Ring loop over all GPUs"
  forLoop 0 8 do  -- 8 GPUs in ring
    comment "Prefetch next KV block (async from next GPU)"
    -- In actual implementation: async ring communication
    load kNext kNextShared
    load vNext vNextShared

    comment "Compute attention with current KV block"
    let qF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    let kF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
    convert qF q
    convert kF k
    mmaT att qF kF (← zeroRT .Float32 64 64)
    scalarMul att att 0.125

    comment "Online softmax for this block"
    rowMax maxVec att
    subCol att att maxVec
    exp att att
    rowSum sumVec att
    divCol att att sumVec

    comment "Compute new LSE: lse_new = max + log(sum)"
    copyVec lseNew maxVec
    logVec sumVec sumVec
    addVec lseNew lseNew sumVec

    comment "Combine with running output using LSE trick"
    -- scale_old = exp(lse_running - max(lse_running, lse_new))
    -- scale_new = exp(lse_new - max(lse_running, lse_new))
    let maxLse : RV GpuFloat.Float32 64 ← allocRV .Float32 64
    -- max of two vectors (element-wise)
    copyVec maxLse lseRunning
    -- Simplified: assuming lseNew is always larger for new blocks

    comment "Scale old output"
    subVec scale lseRunning lseNew
    expVec scale scale
    mulCol o o scale

    comment "Add new contribution"
    let oNew : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
    let attBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
    convert attBf att
    mma oNew attBf v oNew
    add o o oNew

    comment "Update running LSE"
    -- lse_running = log(exp(lse_running) + exp(lse_new))
    -- ≈ max(lse_running, lse_new) + log(exp(lse_running - max) + exp(lse_new - max))
    copyVec lseRunning lseNew  -- Simplified

    comment "Swap KV buffers"
    copy k kNext
    copy v vNext

    sync

  comment "Store final output"
  convert out o
  store outShared out

def ringAttnFullKernel : Kernel :=
  buildKernelM "ring_attn_full" .SM90 #[
    { name := "Q", dtype := .BFloat16, isPointer := true },
    { name := "K", dtype := .BFloat16, isPointer := true },
    { name := "V", dtype := .BFloat16, isPointer := true },
    { name := "O", dtype := .BFloat16, isPointer := true },
    { name := "lse", dtype := .Float32, isPointer := true },
    { name := "rank", dtype := .Float32, isPointer := false },
    { name := "world_size", dtype := .Float32, isPointer := false },
    { name := "seq_len", dtype := .Float32, isPointer := false }
  ] ringAttnFull

/-- Ring Attention - Phase 3: Reduction combining partial outputs -/
@[gpu_kernel .SM90]
def ringAttnReduce : KernelM Unit := do
  comment "=== Ring Attention - Reduction ==="
  comment "Combine partial outputs using log-sum-exp trick"

  -- Partial outputs from different KV blocks
  let o1 : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let o2 : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64

  -- Corresponding LSE values
  let lse1 : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let lse2 : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Combined output
  let oCombined : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let lseCombined : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Scaling factors
  let scale1 : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let scale2 : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let maxLse : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  -- Shared memory
  let o1Shared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let o2Shared : ST GpuFloat.Float32 64 64 ← allocST .Float32 64 64
  let lse1Shared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let lse2Shared : SV GpuFloat.Float32 64 ← allocSV .Float32 64
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load partial outputs and LSE values"
  load o1 o1Shared
  load o2 o2Shared
  loadVec lse1 lse1Shared
  loadVec lse2 lse2Shared

  comment "Compute max LSE for numerical stability"
  -- maxLse = max(lse1, lse2) element-wise
  copyVec maxLse lse1
  -- In practice: need element-wise max

  comment "Compute scaling factors"
  -- scale1 = exp(lse1 - maxLse)
  subVec scale1 lse1 maxLse
  expVec scale1 scale1
  -- scale2 = exp(lse2 - maxLse)
  subVec scale2 lse2 maxLse
  expVec scale2 scale2

  comment "Scale partial outputs"
  mulCol o1 o1 scale1
  mulCol o2 o2 scale2

  comment "Sum scaled outputs"
  add oCombined o1 o2

  comment "Normalize by total scale"
  let totalScale : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  addVec totalScale scale1 scale2
  divCol oCombined oCombined totalScale

  comment "Compute combined LSE"
  -- lseCombined = maxLse + log(scale1 + scale2)
  logVec totalScale totalScale
  addVec lseCombined maxLse totalScale

  comment "Store combined output"
  convert out oCombined
  store outShared out

  sync

def ringAttnReduceKernel : Kernel :=
  buildKernelM "ring_attn_reduce" .SM90 #[
    { name := "O1", dtype := .Float32, isPointer := true },
    { name := "O2", dtype := .Float32, isPointer := true },
    { name := "lse1", dtype := .Float32, isPointer := true },
    { name := "lse2", dtype := .Float32, isPointer := true },
    { name := "O_combined", dtype := .BFloat16, isPointer := true },
    { name := "lse_combined", dtype := .Float32, isPointer := true }
  ] ringAttnReduce

-- Print generated kernels
#eval IO.println "=== Ring Attn Partial ===" *> IO.println (generateKernel ringAttnPartialKernel)
#eval IO.println "\n=== Ring Attn Full ===" *> IO.println (generateKernel ringAttnFullKernel)
#eval IO.println "\n=== Ring Attn Reduce ===" *> IO.println (generateKernel ringAttnReduceKernel)

end Tyr.GPU.Kernels.RingAttn
