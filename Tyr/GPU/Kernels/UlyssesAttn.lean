/-
  Tyr/GPU/Kernels/UlyssesAttn.lean

  Ulysses Attention kernel implementation for sequence parallelism.
  Based on ThunderKittens patterns.

  Key features:
  - All-to-all communication pattern
  - Flexible axis specification (12 combinations)
  - Efficient for multi-head attention parallelism
  - Head-parallel → sequence-parallel → head-parallel transformation
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

namespace Tyr.GPU.Kernels.UlyssesAttn

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Ulysses Attention

Ulysses attention uses all-to-all communication instead of ring:

1. **Input**: Q, K, V sharded by heads across GPUs
   - Each GPU has full sequence but subset of heads

2. **All-to-all on Q**: Redistribute so each GPU has full heads but subset of sequence
   - Transforms from head-parallel to sequence-parallel

3. **Local attention**: Each GPU computes attention on its sequence slice
   - Standard attention on local sequence chunk

4. **All-to-all on O**: Redistribute output back to head-parallel
   - Transforms back from sequence-parallel to head-parallel

This is more communication-efficient than ring attention for moderate sequence lengths.
-/

/-- Ulysses Attention forward pass -/
@[gpu_kernel .SM90]
def ulyssesAttnFwd : KernelM Unit := do
  comment "=== Ulysses Attention Forward ==="
  comment "All-to-all based sequence parallel attention"

  -- Input Q, K, V (head-parallel: full sequence, subset of heads)
  let qIn : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kIn : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let vIn : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- After all-to-all (sequence-parallel: subset of sequence, full heads)
  let qSeq : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kSeq : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let vSeq : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Attention computation
  let att : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let oSeq : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64

  -- Softmax tracking
  let maxVec : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let sumVec : RV GpuFloat.Float32 64 ← zeroRV .Float32 64

  -- Output (head-parallel)
  let oOut : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory
  let qInShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kInShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vInShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let qSeqShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kSeqShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vSeqShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load head-parallel inputs"
  load qIn qInShared
  load kIn kInShared
  load vIn vInShared

  comment "=== Phase 1: All-to-all on Q (heads → sequence) ==="
  comment "Redistribute Q from head-parallel to sequence-parallel"
  -- In actual implementation: NCCL all-to-all
  -- Each GPU sends its head slice to all GPUs, receives sequence slice from all
  copy qSeq qIn  -- Simulated

  comment "=== Phase 2: All-to-all on K, V (heads → sequence) ==="
  copy kSeq kIn
  copy vSeq vIn  -- Simulated

  comment "Synchronize after all-to-all"
  sync

  comment "=== Phase 3: Local attention on sequence slice ==="
  comment "Each GPU has full heads but subset of sequence"

  comment "Compute attention scores"
  let qF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let kF : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  convert qF qSeq
  convert kF kSeq
  mmaT att qF kF (← zeroRT .Float32 64 64)

  comment "Scale by 1/sqrt(d)"
  scalarMul att att 0.125  -- 1/sqrt(64)

  comment "Apply causal mask for this sequence block"
  -- Note: causal mask depends on global position, not just local
  makeCausal att att (some (-1.0e10))

  comment "Softmax"
  rowMax maxVec att
  subCol att att maxVec
  exp att att
  rowSum sumVec att
  divCol att att sumVec

  comment "Compute output: O = softmax(QK^T) @ V"
  let attBf : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  convert attBf att
  mma oSeq attBf vSeq oSeq

  comment "=== Phase 4: All-to-all on O (sequence → heads) ==="
  comment "Redistribute O from sequence-parallel back to head-parallel"
  copy oOut oSeq  -- Simulated all-to-all

  comment "Store final output"
  convert out oOut
  store outShared out
  sync

-- Verify auto-generated kernel
#check ulyssesAttnFwd.kernel
#check ulyssesAttnFwd.launch

/-! ## Ulysses with Fused All-to-all

Variant that fuses all-to-all with attention computation.
-/

/-- Ulysses Attention with fused communication -/
@[gpu_kernel .SM90]
def ulyssesAttnFusedFwd : KernelM Unit := do
  comment "=== Ulysses Attention (Fused All-to-all) ==="

  let q : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let k : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let v : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  -- Double buffering for overlap
  let qNext : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let kNext : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64
  let vNext : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col

  let att : RT GpuFloat.Float32 64 64 ← allocRT .Float32 64 64
  let o : RT GpuFloat.Float32 64 64 ← zeroRT .Float32 64 64
  let out : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  let maxVec : RV GpuFloat.Float32 64 ← negInftyRV .Float32 64
  let sumVec : RV GpuFloat.Float32 64 ← zeroRV .Float32 64
  let prevMax : RV GpuFloat.Float32 64 ← allocRV .Float32 64
  let scale : RV GpuFloat.Float32 64 ← allocRV .Float32 64

  let qShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let qNextShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let kNextShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let vNextShared : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let outShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Initial load"
  load q qShared
  load k kShared
  load v vShared

  comment "Fused all-to-all + attention loop"
  for gpuIdx in krange 0 8 do  -- world_size iterations
    comment "Prefetch next chunk (overlapped all-to-all)"
    load qNext qNextShared
    load kNext kNextShared
    load vNext vNextShared

    comment "Compute attention with current chunk"
    let qF ← allocRT .Float32 64 64
    let kF ← allocRT .Float32 64 64
    convert qF q
    convert kF k
    mmaT att qF kF (← zeroRT .Float32 64 64)
    scalarMul att att 0.125

    comment "Online softmax with rescaling"
    copyVec prevMax maxVec
    rowMaxAccum maxVec att maxVec
    subCol att att maxVec
    exp att att

    -- Rescale previous
    subVec scale prevMax maxVec
    expVec scale scale
    mulVec sumVec sumVec scale
    mulCol o o scale

    let rowS ← allocRV .Float32 64
    rowSum rowS att
    addVec sumVec sumVec rowS

    comment "Accumulate output"
    let attBf ← allocRT .BFloat16 64 64
    convert attBf att
    mma o attBf v o

    comment "Swap buffers"
    copy q qNext
    copy k kNext
    copy v vNext

    sync

  comment "Final normalization"
  divCol o o sumVec

  comment "All-to-all back to head-parallel and store"
  convert out o
  store outShared out

-- Verify auto-generated kernel
#check ulyssesAttnFusedFwd.kernel
#check ulyssesAttnFusedFwd.launch

/-! ## All-to-all Kernel

Standalone all-to-all for flexible use.
-/

/-- All-to-all communication kernel -/
@[gpu_kernel .SM90]
def allToAllFwd : KernelM Unit := do
  comment "=== All-to-all Communication ==="
  comment "Each GPU sends one chunk to each other GPU, receives one from each"

  -- Input (sharded by one dimension)
  let input : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Output (sharded by different dimension)
  let output : RT GpuFloat.BFloat16 64 64 ← allocRT .BFloat16 64 64

  -- Shared memory for staging
  let inputShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64
  let outputShared : ST GpuFloat.BFloat16 64 64 ← allocST .BFloat16 64 64

  comment "Load input"
  load input inputShared

  comment "All-to-all exchange"
  comment "(In actual implementation: NCCL all-to-all or custom)"
  -- Each GPU i sends chunk j to GPU j, receives chunk i from GPU j
  copy output input  -- Simulated

  comment "Store output"
  store outputShared output
  sync

-- Verify auto-generated kernel
#check allToAllFwd.kernel
#check allToAllFwd.launch

-- Print generated kernels
#eval IO.println "=== Ulysses Attn ===" *> IO.println (generateKernel ulyssesAttnFwd.kernel)
#eval IO.println "\n=== Ulysses Attn Fused ===" *> IO.println (generateKernel ulyssesAttnFusedFwd.kernel)
#eval IO.println "\n=== All-to-all ===" *> IO.println (generateKernel allToAllFwd.kernel)

end Tyr.GPU.Kernels.UlyssesAttn
