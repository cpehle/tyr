/-
  Tyr/GPU/Kernels/UlyssesAttnBwd.lean

  Ulysses Attention backward kernel implementation.
  Wraps FlashAttention backward with All-to-All communication.
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
import Tyr.GPU.Kernels.FlashAttnBwd

namespace Tyr.GPU.Kernels.UlyssesAttn

open Tyr.GPU
open Tyr.GPU.Codegen
open Tyr.GPU.Kernels (flashAttnBwd flashAttnBwdPrep)

/-! ## Ulysses Attention Backward

Reverse of the forward pass:
1. All-to-all on dO (heads → sequence)
2. Local FlashAttention Backward
3. All-to-all on dQ, dK, dV (sequence → heads)
-/

/-- Ulysses Attention Backward Kernel -/
@[gpu_kernel .SM90]
def ulyssesAttnBwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32) (D_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32) (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (seq_len : KVal UInt64) (head_dim : KVal UInt64)
    : KernelM Unit := do
  
  let tileSize : Nat := 64
  let numKvBlocks : Nat := 4
  
  comment "=== Ulysses Attention Backward ==="

  -- 1. All-to-all on dO: Head-parallel -> Sequence-parallel
  comment "Phase 1: Redistribute dO (Heads -> Sequence)"
  
  let dO : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let dOShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  
  -- Simulate A2A: Load local dO part, then 'transmit' to peers
  -- In reality, this is a collective op.
  load dO dOShared 
  
  -- For simulation of A2A, we'd store to a multimem buffer.
  multimemStore dOShared dO
  sync
  
  -- After A2A, we have dO for the *sequence* slice.
  -- We assume dO now contains the correct slice.
  
  comment "Phase 2: Local FlashAttention Backward"
  
  -- Local Q, K, V are assumed to be available in Sequence Parallel layout 
  -- (e.g. from saved activation or re-A2A).
  let q : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let k : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let v : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
  
  -- Gradient Accumulators
  let dQ : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize
  let dK : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize
  let dV : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize
  
  let qShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let kShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let vShared : ST GpuFloat.BFloat16 tileSize tileSize .Col ← allocST .BFloat16 tileSize tileSize .Col
  
  -- LSE and D vectors
  let lseVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let dVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let lseShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  let dVecShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  
  loadVec lseVec lseShared
  loadVec dVec dVecShared
  
  -- Load Q (long resident)
  load q qShared

  -- Intermediate vars
  let s : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let p : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dP : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dS : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dSBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let rowMaxVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  for blkIdx in krange 0 numKvBlocks do
    load k kShared
    load v vShared
    
    -- S = Q @ K^T
    mmaT s q k s
    makeCausal s s (some (-1e10))
    
    -- P = exp(S - L)
    rowMax rowMaxVec s -- Actually LSE is preloaded
    subCol s s lseVec
    exp p s
    
    -- dP = dO @ V^T
    let vRow : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v
    mmaT dP dO vRow dP
    
    -- dS = P * (dP - D)
    subCol dP dP dVec
    mul dS p dP
    makeCausal dS dS (some 0.0)
    convert dSBf16 dS
    
    -- dQ += dS @ K
    let kCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSBf16 kCol dQ
    
    -- dK += dS^T @ Q
    let dST : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    transpose dST dSBf16
    let qCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dK dST qCol dK
    
    -- dV += P^T @ dO
    let pT : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    let pBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    convert pBf16 p
    transpose pT pBf16
    let dOCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout dOCol dO
    mma dV pT dOCol dV
    
    sync

  comment "Phase 3: Redistribute Gradients (All-to-All: Seq -> Heads)"
  
  -- dQ, dK, dV are currently Sequence Parallel.
  -- A2A them back to Head Parallel.
  
  let dQShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  let dKShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  let dVShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  
  store dQShared dQ
  store dKShared dK
  store dVShared dV
  
  multimemStore dQShared dQ -- Simulate A2A
  multimemStore dKShared dK
  multimemStore dVShared dV
  
  sync
  
  -- Final store to global (Head Parallel pointers)
  -- storeAdd used for atomic accumulation if necessary
  storeAdd dQShared dQ
  storeAdd dKShared dK
  storeAdd dVShared dV


-- Verify
#check ulyssesAttnBwd.kernel

-- Generate
#eval IO.println "=== Ulysses Attention Backward ===" *>
      IO.println (generateKernel ulyssesAttnBwd.kernel)

end Tyr.GPU.Kernels.UlyssesAttn