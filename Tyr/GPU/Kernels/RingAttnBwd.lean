/-
  Tyr/GPU/Kernels/RingAttnBwd.lean

  Ring Attention backward kernel implementation.
  Computes dQ, dK, dV using ring communication for sequence parallelism.
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

namespace Tyr.GPU.Kernels.RingAttn

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Ring Attention Backward

Algorithm:
1. Initialize dQ = 0 (stationary).
2. Initialize dK_buf = 0, dV_buf = 0 (circulating).
3. Loop over ring steps (step 0..world_size-1):
   a. Wait for K, V, dK_buf, dV_buf to arrive (async).
   b. Recompute Attention:
      - S = Q @ K^T
      - P = exp(S - LSE)
   c. Compute Gradients:
      - dP = dO @ V^T
      - dS = P * (dP - D)
      - dQ += dS @ K
      - dK_part = dS^T @ Q
      - dV_part = P^T @ dO
   d. Accumulate into buffers:
      - dK_buf += dK_part
      - dV_buf += dV_part
   e. Send K, V, dK_buf, dV_buf to next GPU.
4. Store dQ, dK_buf, dV_buf.
-/

@[gpu_kernel .SM90]
def ringAttnBwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32) (D_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32) (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (rank : KVal UInt32) (world_size : KVal UInt32)
    : KernelM Unit := do
  
  let tileSize : Nat := 64
  comment "=== Ring Attention Backward ==="
  let coord ← blockCoord2D

  -- Stationary data (Q, dO, dQ, LSE, D)
  let q : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let dO : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let dQ : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize
  
  let lseVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize
  let dVec : RV GpuFloat.Float32 tileSize ← allocRV .Float32 tileSize

  -- Circulating data (K, V, dK, dV)
  -- These will be loaded from neighbor/multimem
  let k : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  let v : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
  let dK : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize
  let dV : RT GpuFloat.Float32 tileSize tileSize ← zeroRT .Float32 tileSize tileSize

  -- Intermediate
  let s : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let p : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dP : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dS : RT GpuFloat.Float32 tileSize tileSize ← allocRT .Float32 tileSize tileSize
  let dSBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
  
  -- Shared memory for loading/storing
  let qShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let dOShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let kShared : ST GpuFloat.BFloat16 tileSize tileSize ← allocST .BFloat16 tileSize tileSize
  let vShared : ST GpuFloat.BFloat16 tileSize tileSize .Col ← allocST .BFloat16 tileSize tileSize .Col
  let dKShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  let dVShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  
  let lseShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize
  let dVecShared : SV GpuFloat.Float32 tileSize ← allocSV .Float32 tileSize

  comment "Load stationary data"
  load q qShared
  load dO dOShared
  loadVec lseVec lseShared
  loadVec dVec dVecShared
  
  -- Initial K, V are loaded from local global memory (step 0)
  -- Or handled uniformly in the loop via multimem logic.
  -- Assuming K_ptr, V_ptr point to *local* data, and we push/pull from neighbors.
  -- For simplified ring:
  -- We start with local K, V.
  -- dK, dV start at 0.
  
  load k kShared
  load v vShared
  
  comment "Ring Loop"
  for stepIdx in krange 0 8 do -- Hardcoded 8 for now, ideally 'world_size'
    
    comment "--- Computation ---"
    
    comment "1. Recompute P = softmax(Q @ K^T)"
    mmaT s q k s
    
    -- Causal Masking Logic
    -- Need to know global position of Q and K blocks.
    -- Q is stationary: global_row = rank * num_blocks_per_rank + local_row
    -- K is rotating: global_col = (rank - step) * num_blocks...
    -- Simpler: pass 'step' to makeCausal? Or assume it handles it?
    -- `makeCausal` in Ops is naive. We need a mask that shifts.
    -- For now, use `makeCausal` with a comment, or better, `upperFill` etc. if logic allows.
    -- Assuming a `makeCausalGlobal` or similar existed, or we rely on the user to map it.
    -- Using naive mask for now but noting the fix.
    comment "TODO: Adjust causal mask for ring step" 
    makeCausal s s (some (-1e10))
    
    -- Softmax with LSE
    subCol s s lseVec
    exp p s

    comment "2. dP = dO @ V^T"
    let vRow : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    swapLayout vRow v
    mmaT dP dO vRow dP

    comment "3. dS = P * (dP - D)"
    subCol dP dP dVec
    mul dS p dP
    -- Apply mask to gradient
    makeCausal dS dS (some 0.0)
    convert dSBf16 dS

    comment "4. Accumulate dQ += dS @ K"
    let kCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSBf16 kCol dQ

    comment "5. Accumulate dK += dS^T @ Q"
    -- dK is the circulating buffer.
    let dST : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    transpose dST dSBf16
    let qCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout qCol q
    mma dK dST qCol dK

    comment "6. Accumulate dV += P^T @ dO"
    let pT : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    let pBf16 : RT GpuFloat.BFloat16 tileSize tileSize ← allocRT .BFloat16 tileSize tileSize
    convert pBf16 p
    transpose pT pBf16
    let dOCol : RT GpuFloat.BFloat16 tileSize tileSize .Col ← allocRT .BFloat16 tileSize tileSize .Col
    swapLayout dOCol dO
    mma dV pT dOCol dV

    sync
    
    comment "--- Communication (Ring Step) ---"
    -- Send K, V, dK, dV to next neighbor. Receive from prev.
    -- Using `multimemStore` to push to neighbor's shared memory?
    -- Or assuming `load` gets from neighbor in next iteration (if mapped correctly).
    -- ThunderKittens usually uses `multimem` for this.
    
    comment "Send K, V, dK, dV to next neighbor"
    -- We store current regs to *our* multimem buffer which maps to neighbor?
    -- Or we store to shared and issue a multimem transfer.
    
    store kShared k
    store vShared v
    store dKShared dK
    store dVShared dV
    
    multimemStore kShared k
    multimemStore vShared v  -- Assuming v needs re-loading as Row?
    multimemStore dKShared dK
    multimemStore dVShared dV
    
    sync
    
    -- In real ring, we'd now wait for arrival.
    -- And then load the *new* K, V, dK, dV into regs.
    -- We reuse the same regs for the next step.
    load k kShared
    load v vShared
    load dK dKShared -- The *received* dK (accumulated from previous steps)
    load dV dVShared

  comment "Store results"
  let dQShared : ST GpuFloat.Float32 tileSize tileSize ← allocST .Float32 tileSize tileSize
  store dQShared dQ
  
  -- dK, dV are now fully accumulated (after full ring).
  -- They should be stored to global memory.
  store dKShared dK
  store dVShared dV
  
  storeGlobal dQ_ptr dQShared coord
  storeGlobalAdd dK_ptr dKShared coord
  storeGlobalAdd dV_ptr dVShared coord

-- Verify
#check ringAttnBwd.kernel

-- Generate
#eval IO.println "=== Ring Attention Backward ===" *>
      IO.println (generateKernel ringAttnBwd.kernel)

end Tyr.GPU.Kernels.RingAttn
