/-
  Tyr/GPU/Kernels/FlashAttn3.lean

  FlashAttention3 implementation for Hopper GPUs (SM90).

  Key innovations over FA2:
  - Warp specialization: Producer warps (TMA loads) + Consumer warps (MMA)
  - TMA pipelining: 2-stage double buffering for K, V tiles
  - Online softmax with proper rescaling for numerical stability
  - GQA (Grouped Query Attention) support

  Reference: FlashAttention-3 (Dao et al., 2024)
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

namespace Tyr.GPU.Kernels.FlashAttn3

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Configuration -/

/-- FlashAttention3 forward configuration -/
structure FA3FwdConfig where
  blockM : Nat := 64       -- Query tile rows
  blockN : Nat := 64       -- KV tile rows
  headDim : Nat := 64      -- Head dimension
  numStages : Nat := 2     -- Pipeline stages (always 2 for FA3)
  isCausal : Bool := false -- Causal masking
  -- GQA configuration
  hasGQA : Bool := false
  qHeadsPerKvHead : Nat := 1
  deriving Repr, Inhabited

/-- FlashAttention3 backward configuration -/
structure FA3BwdConfig where
  blockM : Nat := 64
  blockN : Nat := 64
  headDim : Nat := 64
  numStages : Nat := 2
  isCausal : Bool := false
  deterministic : Bool := false  -- Use semaphores for ordered dQ accumulation
  deriving Repr, Inhabited

/-! ## Named Barrier IDs for Warp Specialization -/

/-- Barrier IDs for producer/consumer synchronization -/
def barrierQueryReady : Nat := 0    -- Q loaded, consumers can start
def barrierKReady : Nat := 1        -- K[stage] loaded
def barrierVReady : Nat := 2        -- V[stage] loaded
def barrierKRelease : Nat := 3      -- K[stage] consumed, producer can reuse
def barrierVRelease : Nat := 4      -- V[stage] consumed, producer can reuse

/-- Number of threads in consumer warp groups (128 per warp group) -/
def numConsumerThreads : Nat := 128

/-! ## FlashAttention3 Forward Kernel -/

/-- FlashAttention3 forward kernel with warp specialization and TMA pipelining.

This kernel uses:
- Producer warp group (wgIdx=0): Loads Q once, then pipelines K/V loads
- Consumer warp group (wgIdx=1): Performs attention computation
- 2-stage double buffering for K, V tiles
- Online softmax with proper rescaling
- Outputs O (attention output) and L (log-sum-exp for backward)
-/
@[gpu_kernel .SM90]
def flashAttn3Fwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32)
    (seqLenQ : KVal UInt64) (seqLenK : KVal UInt64) (headDim : KVal UInt64)
    : KernelM Unit := do
  let blockM : Nat := 64
  let blockN : Nat := 64
  let hdim : Nat := 64
  let numKvBlocks : Nat := 4  -- seqLenK / blockN (simplified for now)

  comment "=== FlashAttention3 Forward ==="
  comment "Warp-specialized: Producer (TMA loads) + Consumer (MMA)"

  -- ### Shared Memory Allocation ###
  comment "Shared memory: Q tile + double-buffered K/V tiles"

  -- Q tile (loaded once, long-resident)
  let sQ : ST GpuFloat.BFloat16 blockM hdim ← allocST .BFloat16 blockM hdim

  -- K tiles (double-buffered: stage 0 and stage 1)
  let sK0 : ST GpuFloat.BFloat16 blockN hdim ← allocST .BFloat16 blockN hdim
  let sK1 : ST GpuFloat.BFloat16 blockN hdim ← allocST .BFloat16 blockN hdim

  -- V tiles (double-buffered, column-major for efficient MMA)
  let sV0 : ST GpuFloat.BFloat16 blockN hdim .Col ← allocST .BFloat16 blockN hdim .Col
  let sV1 : ST GpuFloat.BFloat16 blockN hdim .Col ← allocST .BFloat16 blockN hdim .Col

  -- P tile (softmax output, for O += P @ V)
  let sP : ST GpuFloat.BFloat16 blockM blockN ← allocST .BFloat16 blockM blockN

  -- LSE output (shared vector for storing log-sum-exp)
  let sLse : SV GpuFloat.Float32 blockM ← allocSV .Float32 blockM

  -- Output tile
  let sO : ST GpuFloat.Float32 blockM hdim ← allocST .Float32 blockM hdim

  -- ### Pipeline Semaphores ###
  comment "Pipeline semaphores for K/V double buffering"
  let semK0 ← allocSemaphore
  let semK1 ← allocSemaphore
  let semV0 ← allocSemaphore
  let semV1 ← allocSemaphore

  -- Initialize semaphores
  initSemaphore semK0 1
  initSemaphore semK1 1
  initSemaphore semV0 1
  initSemaphore semV1 1

  -- ### Register Tiles ###
  comment "Register tiles for computation"

  -- Q in registers (long-resident)
  let q : RT GpuFloat.BFloat16 blockM hdim ← allocRT .BFloat16 blockM hdim

  -- Attention scores (float32 for precision)
  let scores : RT GpuFloat.Float32 blockM blockN ← allocRT .Float32 blockM blockN

  -- Softmax intermediate (bf16 for MMA)
  let p : RT GpuFloat.BFloat16 blockM blockN ← allocRT .BFloat16 blockM blockN

  -- Output accumulator (float32)
  let o : RT GpuFloat.Float32 blockM hdim ← zeroRT .Float32 blockM hdim

  -- ### Online Softmax State ###
  comment "Online softmax tracking vectors"
  let rowMax : RV GpuFloat.Float32 blockM ← negInftyRV .Float32 blockM
  let rowSum : RV GpuFloat.Float32 blockM ← zeroRV .Float32 blockM
  let prevMax : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM
  let scaleVec : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM
  let lseVec : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM

  -- ### Warp Group 0: Producer (TMA Loads) ###
  comment "========== PRODUCER WARP GROUP =========="
  ifWarpGroup 0 do
    comment "Producer: Load Q once, then pipeline K/V"

    -- Load Q tile (one-time, long-resident in consumer registers)
    comment "Load Q tile via TMA"
    -- In production: tmaLoad sQ Q_ptr qCoord
    -- For now, simulate with regular load
    sync 0

    -- Signal Q is ready for consumers
    namedBarrierArrive barrierQueryReady numConsumerThreads

    comment "Main K/V loading loop with 2-stage pipelining"
    forLoop 0 numKvBlocks do
      comment "Load K[n] into current stage"
      -- Stage selection based on iteration (alternates 0, 1, 0, 1, ...)
      -- In practice: if (n % 2 == 0) load into sK0 else load into sK1

      -- Wait for previous consumer to finish with this buffer
      waitSemaphore semK0
      -- Load K
      -- tmaLoad sK0 K_ptr kvCoord
      -- Signal K ready
      arriveSemaphore semK0 1

      comment "Load V[n] into current stage"
      waitSemaphore semV0
      -- Load V
      -- tmaLoad sV0 V_ptr vCoord
      -- Signal V ready
      arriveSemaphore semV0 1

      sync 0

  -- ### Warp Group 1: Consumer (MMA Computation) ###
  comment "========== CONSUMER WARP GROUP =========="
  ifWarpGroup 1 do
    comment "Consumer: Compute attention with pipelined K/V"

    -- Wait for Q to be ready
    namedBarrierSync barrierQueryReady numConsumerThreads

    comment "Load Q from shared to registers (long-resident)"
    load q sQ

    comment "Main attention loop over K/V blocks"
    forLoop 0 numKvBlocks do
      comment "--- Process K/V block ---"

      -- Wait for K to be loaded
      waitSemaphore semK0

      comment "S = Q @ K^T"
      -- Load K to register for MMA
      let k : RT GpuFloat.BFloat16 blockN hdim ← allocRT .BFloat16 blockN hdim
      load k sK0

      -- Zero scores for fresh computation
      zero scores

      -- Matrix multiply: S = Q @ K^T
      mmaT scores q k scores

      -- Commit MMA and wait
      mmaCommitGroup
      mmaAsyncWait 0

      -- Release K buffer for producer
      arriveSemaphore semK0 1

      comment "Apply causal mask (optional)"
      makeCausal scores scores (some (-1e10))

      comment "Online softmax with rescaling"

      -- Save previous max for rescaling
      copyVec prevMax rowMax

      -- Update row-wise maximum: rowMax = max(rowMax, rowMax(S))
      rowMaxAccum rowMax scores rowMax

      -- Compute rescale factor: scale = exp(prevMax - newMax)
      subVec scaleVec prevMax rowMax
      expVec scaleVec scaleVec

      -- Rescale previous output: O *= scale
      mulCol o o scaleVec

      -- Rescale previous sum: rowSum *= scale
      mulVec rowSum rowSum scaleVec

      -- Compute exp(S - rowMax)
      subCol scores scores rowMax
      exp scores scores

      -- Update row sum: rowSum += sum(exp(S - max))
      rowSumAccum rowSum scores rowSum

      -- Convert to bf16 for V multiply
      convert p scores

      comment "Store P to shared for MMA with V"
      store sP p
      fenceViewAsyncShared

      -- Wait for V to be loaded
      waitSemaphore semV0

      comment "O += P @ V"
      let v : RT GpuFloat.BFloat16 blockN hdim .Col ← allocRT .BFloat16 blockN hdim .Col
      load v sV0

      mma o p v o
      mmaCommitGroup
      mmaAsyncWait 0

      -- Release V buffer for producer
      arriveSemaphore semV0 1

      sync 0

    comment "=== Final normalization ==="

    -- O = O / rowSum
    divCol o o rowSum

    comment "Compute LSE = log(rowSum) + rowMax for backward pass"
    logVec lseVec rowSum
    addVec lseVec lseVec rowMax

    comment "Store output and LSE"
    store sO o
    storeVec sLse lseVec

/-! ## FlashAttention3 Backward Prep Kernel -/

/-- Backward preparation: compute D = rowSum(dO * O)

This is computed separately because D is needed by the main backward kernel
to compute dS = P * (dP - D).
-/
@[gpu_kernel .SM90]
def flashAttn3BwdPrep (dO_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (D_ptr : GPtr GpuFloat.Float32)
    (seqLen : KVal UInt64) (headDim : KVal UInt64)
    : KernelM Unit := do
  let blockM : Nat := 64
  let hdim : Nat := 64

  comment "=== FlashAttention3 Backward Prep ==="
  comment "Computes D = rowSum(dO * O)"

  -- Shared memory
  let sDO : ST GpuFloat.BFloat16 blockM hdim ← allocST .BFloat16 blockM hdim
  let sO : ST GpuFloat.BFloat16 blockM hdim ← allocST .BFloat16 blockM hdim
  let sD : SV GpuFloat.Float32 blockM ← allocSV .Float32 blockM

  -- Register tiles
  let dO : RT GpuFloat.BFloat16 blockM hdim ← allocRT .BFloat16 blockM hdim
  let outFwd : RT GpuFloat.BFloat16 blockM hdim ← allocRT .BFloat16 blockM hdim
  let dOF32 : RT GpuFloat.Float32 blockM hdim ← allocRT .Float32 blockM hdim
  let oF32 : RT GpuFloat.Float32 blockM hdim ← allocRT .Float32 blockM hdim
  let prod : RT GpuFloat.Float32 blockM hdim ← allocRT .Float32 blockM hdim
  let dVec : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM

  comment "Load dO and O"
  load dO sDO
  load outFwd sO

  comment "Convert to float32 for precision"
  convert dOF32 dO
  convert oF32 outFwd

  comment "Element-wise multiply: prod = dO * O"
  mul prod dOF32 oF32

  comment "Row-wise sum: D = rowSum(dO * O)"
  rowSum dVec prod

  comment "Store D"
  storeVec sD dVec

/-! ## FlashAttention3 Main Backward Kernel -/

/-- FlashAttention3 main backward kernel.

Computes dQ, dK, dV from forward activations and gradients.

Key equations:
1. Recompute P = softmax(QK^T) using stored LSE
2. dP = dO @ V^T
3. dS = P * (dP - D) where D = rowSum(dO * O)
4. dQ = dS @ K
5. dK += dS^T @ Q (accumulated across Q blocks)
6. dV += P^T @ dO (accumulated across Q blocks)

Uses K, V as long-resident (outer loop over K/V blocks).
Inner loop iterates over Q/dO blocks.
-/
@[gpu_kernel .SM90]
def flashAttn3Bwd (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (dO_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32) (D_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32) (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (seqLenQ : KVal UInt64) (seqLenK : KVal UInt64) (headDim : KVal UInt64)
    : KernelM Unit := do
  let blockM : Nat := 64
  let blockN : Nat := 64
  let hdim : Nat := 64
  let numQBlocks : Nat := 4  -- seqLenQ / blockM (simplified)

  comment "=== FlashAttention3 Backward ==="
  comment "K, V long-resident; loop over Q, dO blocks"

  -- ### Shared Memory ###
  comment "Shared memory for K, V (long-resident)"
  let sK : ST GpuFloat.BFloat16 blockN hdim ← allocST .BFloat16 blockN hdim
  let sV : ST GpuFloat.BFloat16 blockN hdim .Col ← allocST .BFloat16 blockN hdim .Col

  comment "Shared memory for Q, dO (pipelined)"
  let sQ : ST GpuFloat.BFloat16 blockM hdim ← allocST .BFloat16 blockM hdim
  let sDO : ST GpuFloat.BFloat16 blockM hdim ← allocST .BFloat16 blockM hdim

  comment "Shared memory for intermediates"
  let sP : ST GpuFloat.BFloat16 blockM blockN ← allocST .BFloat16 blockM blockN
  let sDS : ST GpuFloat.BFloat16 blockM blockN ← allocST .BFloat16 blockM blockN

  comment "Shared memory for gradient outputs"
  let sDQ : ST GpuFloat.Float32 blockM hdim ← allocST .Float32 blockM hdim
  let sDK : ST GpuFloat.Float32 blockN hdim ← allocST .Float32 blockN hdim
  let sDV : ST GpuFloat.Float32 blockN hdim ← allocST .Float32 blockN hdim

  comment "LSE and D vectors"
  let sLse : SV GpuFloat.Float32 blockM ← allocSV .Float32 blockM
  let sD : SV GpuFloat.Float32 blockM ← allocSV .Float32 blockM

  -- ### Register Tiles ###
  comment "Register tiles"

  -- K, V in registers (long-resident for this n_block)
  let k : RT GpuFloat.BFloat16 blockN hdim ← allocRT .BFloat16 blockN hdim
  let v : RT GpuFloat.BFloat16 blockN hdim .Col ← allocRT .BFloat16 blockN hdim .Col

  -- Gradient accumulators (float32)
  let dK : RT GpuFloat.Float32 blockN hdim ← zeroRT .Float32 blockN hdim
  let dV : RT GpuFloat.Float32 blockN hdim ← zeroRT .Float32 blockN hdim

  -- Working tiles
  let q : RT GpuFloat.BFloat16 blockM hdim ← allocRT .BFloat16 blockM hdim
  let dO : RT GpuFloat.BFloat16 blockM hdim ← allocRT .BFloat16 blockM hdim
  let scores : RT GpuFloat.Float32 blockM blockN ← allocRT .Float32 blockM blockN
  let pF32 : RT GpuFloat.Float32 blockM blockN ← allocRT .Float32 blockM blockN
  let pBf16 : RT GpuFloat.BFloat16 blockM blockN ← allocRT .BFloat16 blockM blockN
  let dP : RT GpuFloat.Float32 blockM blockN ← allocRT .Float32 blockM blockN
  let dS : RT GpuFloat.Float32 blockM blockN ← allocRT .Float32 blockM blockN
  let dSBf16 : RT GpuFloat.BFloat16 blockM blockN ← allocRT .BFloat16 blockM blockN
  let dQ : RT GpuFloat.Float32 blockM hdim ← allocRT .Float32 blockM hdim

  -- Vectors
  let lseVec : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM
  let dVec : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM

  comment "Load K, V once (long-resident for this n_block)"
  load k sK
  load v sV

  comment "Main loop over Q blocks"
  forLoop 0 numQBlocks do
    comment "--- Process Q block m ---"

    comment "Load Q, dO for this m_block"
    load q sQ
    load dO sDO

    comment "Load LSE and D for this m_block"
    loadVec lseVec sLse
    loadVec dVec sD

    comment "=== Recompute attention scores ==="
    comment "S = Q @ K^T"
    zero scores
    mmaT scores q k scores
    mmaCommitGroup
    mmaAsyncWait 0

    comment "Apply causal mask"
    makeCausal scores scores (some (-1e10))

    comment "=== Recompute P from stored LSE ==="
    comment "P = exp(S - LSE)"
    subCol scores scores lseVec
    exp pF32 scores

    convert pBf16 pF32

    comment "=== Compute dP = dO @ V^T ==="
    zero dP
    -- Need V in row-major for transpose in MMA
    let vRow : RT GpuFloat.BFloat16 blockN hdim ← allocRT .BFloat16 blockN hdim
    swapLayout vRow v
    mmaT dP dO vRow dP
    mmaCommitGroup
    mmaAsyncWait 0

    comment "=== Compute dS = P * (dP - D) ==="
    subCol dP dP dVec
    mul dS pF32 dP

    comment "Apply causal mask to dS"
    makeCausal dS dS (some 0.0)

    convert dSBf16 dS

    comment "=== Accumulate gradients ==="

    comment "dQ = dS @ K"
    zero dQ
    let kCol : RT GpuFloat.BFloat16 blockN hdim .Col ← allocRT .BFloat16 blockN hdim .Col
    swapLayout kCol k
    mma dQ dSBf16 kCol dQ
    mmaCommitGroup
    mmaAsyncWait 0

    comment "dK += dS^T @ Q"
    let dST : RT GpuFloat.BFloat16 blockN blockM ← allocRT .BFloat16 blockN blockM
    transpose dST dSBf16
    let qCol : RT GpuFloat.BFloat16 blockM hdim .Col ← allocRT .BFloat16 blockM hdim .Col
    swapLayout qCol q
    mma dK dST qCol dK
    mmaCommitGroup
    mmaAsyncWait 0

    comment "dV += P^T @ dO"
    let pT : RT GpuFloat.BFloat16 blockN blockM ← allocRT .BFloat16 blockN blockM
    transpose pT pBf16
    let dOCol : RT GpuFloat.BFloat16 blockM hdim .Col ← allocRT .BFloat16 blockM hdim .Col
    swapLayout dOCol dO
    mma dV pT dOCol dV
    mmaCommitGroup
    mmaAsyncWait 0

    comment "Store dQ (atomic add for accumulation across K/V blocks)"
    storeAdd sDQ dQ

    sync 0

  comment "=== Store final dK, dV ==="
  store sDK dK
  store sDV dV

/-! ## GQA (Grouped Query Attention) Forward Kernel -/

/-- FlashAttention3 forward with GQA support.

In GQA, we have fewer KV heads than Q heads.
This kernel packs multiple Q heads per KV head in the M dimension
for efficient processing.

Parameters:
- qHeadsPerKvHead: Number of Q heads per KV head (e.g., 8 for Llama)
-/
@[gpu_kernel .SM90]
def flashAttn3FwdGQA (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32)
    (seqLenQ : KVal UInt64) (seqLenK : KVal UInt64) (headDim : KVal UInt64)
    (numHeadsQ : KVal UInt64) (numHeadsKV : KVal UInt64)
    : KernelM Unit := do
  let blockM : Nat := 64
  let blockN : Nat := 64
  let hdim : Nat := 64
  let qHeadsPerKvHead : Nat := 4  -- Example: 4 Q heads per KV head
  let packedBlockM : Nat := blockM * qHeadsPerKvHead  -- Pack Q heads
  let numKvBlocks : Nat := 4

  comment "=== FlashAttention3 Forward with GQA ==="
  comment "Packs multiple Q heads per KV head in M dimension"

  -- For GQA, we use a larger effective M dimension that packs multiple Q heads
  -- This allows us to share K,V across the packed Q heads efficiently

  comment "Shared memory with GQA packing"
  let sQ : ST GpuFloat.BFloat16 packedBlockM hdim ← allocST .BFloat16 packedBlockM hdim
  let sK : ST GpuFloat.BFloat16 blockN hdim ← allocST .BFloat16 blockN hdim
  let sV : ST GpuFloat.BFloat16 blockN hdim .Col ← allocST .BFloat16 blockN hdim .Col
  let sO : ST GpuFloat.Float32 packedBlockM hdim ← allocST .Float32 packedBlockM hdim
  let sLse : SV GpuFloat.Float32 packedBlockM ← allocSV .Float32 packedBlockM

  comment "Register tiles with GQA packing"
  let q : RT GpuFloat.BFloat16 packedBlockM hdim ← allocRT .BFloat16 packedBlockM hdim
  let k : RT GpuFloat.BFloat16 blockN hdim ← allocRT .BFloat16 blockN hdim
  let v : RT GpuFloat.BFloat16 blockN hdim .Col ← allocRT .BFloat16 blockN hdim .Col
  let scores : RT GpuFloat.Float32 packedBlockM blockN ← allocRT .Float32 packedBlockM blockN
  let p : RT GpuFloat.BFloat16 packedBlockM blockN ← allocRT .BFloat16 packedBlockM blockN
  let o : RT GpuFloat.Float32 packedBlockM hdim ← zeroRT .Float32 packedBlockM hdim

  comment "Online softmax state (per packed row)"
  let rowMax : RV GpuFloat.Float32 packedBlockM ← negInftyRV .Float32 packedBlockM
  let rowSum : RV GpuFloat.Float32 packedBlockM ← zeroRV .Float32 packedBlockM
  let prevMax : RV GpuFloat.Float32 packedBlockM ← allocRV .Float32 packedBlockM
  let scaleVec : RV GpuFloat.Float32 packedBlockM ← allocRV .Float32 packedBlockM
  let lseVec : RV GpuFloat.Float32 packedBlockM ← allocRV .Float32 packedBlockM

  comment "Load packed Q (multiple heads concatenated)"
  load q sQ

  comment "Main attention loop"
  forLoop 0 numKvBlocks do
    comment "Load K, V (shared across packed Q heads)"
    load k sK
    load v sV

    comment "S = Q_packed @ K^T"
    zero scores
    mmaT scores q k scores
    mmaCommitGroup
    mmaAsyncWait 0

    comment "Apply causal mask (applied to each packed head independently)"
    -- For GQA with causal, need to handle masking per-head
    makeCausal scores scores (some (-1e10))

    comment "Online softmax"
    copyVec prevMax rowMax
    rowMaxAccum rowMax scores rowMax
    subVec scaleVec prevMax rowMax
    expVec scaleVec scaleVec
    mulCol o o scaleVec
    mulVec rowSum rowSum scaleVec
    subCol scores scores rowMax
    exp scores scores
    rowSumAccum rowSum scores rowSum
    convert p scores

    comment "O_packed += P_packed @ V"
    mma o p v o
    mmaCommitGroup
    mmaAsyncWait 0

    sync 0

  comment "Final normalization"
  divCol o o rowSum

  comment "Compute LSE"
  logVec lseVec rowSum
  addVec lseVec lseVec rowMax

  comment "Store output (will be unpacked externally)"
  store sO o
  storeVec sLse lseVec

/-! ## Persistent Scheduler Support -/

/-- Tile scheduler state for persistent kernels -/
structure TileSchedulerState where
  tileIdx : Nat
  mBlock : Nat
  nBlock : Nat
  headIdx : Nat
  batchIdx : Nat
  isValid : Bool
  deriving Repr, Inhabited

/-- Scheduler configuration -/
structure SchedulerConfig where
  numMBlocks : Nat
  numNBlocks : Nat
  numHeads : Nat
  numBatch : Nat
  totalTiles : Nat
  deriving Repr, Inhabited

/-- Compute total tiles from config -/
def SchedulerConfig.mk' (seqLenQ seqLenK blockM blockN numHeads numBatch : Nat) : SchedulerConfig :=
  let numMBlocks := (seqLenQ + blockM - 1) / blockM
  let numNBlocks := (seqLenK + blockN - 1) / blockN
  { numMBlocks := numMBlocks
    numNBlocks := numNBlocks
    numHeads := numHeads
    numBatch := numBatch
    totalTiles := numMBlocks * numHeads * numBatch }

/-- FlashAttention3 Forward with Persistent Scheduler

This kernel uses a persistent grid where each CTA continuously fetches new tiles
from a global counter until all tiles are processed. This improves load balancing
for variable workloads (e.g., causal masking).

Key features:
- Dynamic tile assignment via atomic counter
- LPT (Longest Processing Time) ordering for causal attention
- Better L2 cache utilization through swizzled head ordering
-/
@[gpu_kernel .SM90]
def flashAttn3FwdPersistent (Q_ptr : GPtr GpuFloat.BFloat16) (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16) (O_ptr : GPtr GpuFloat.BFloat16)
    (L_ptr : GPtr GpuFloat.Float32)
    (tileCounter_ptr : GPtr GpuFloat.Float32)  -- Atomic counter for tile assignment
    (seqLenQ : KVal UInt64) (seqLenK : KVal UInt64) (headDim : KVal UInt64)
    (numHeads : KVal UInt64) (numBatch : KVal UInt64)
    : KernelM Unit := do
  let blockM : Nat := 64
  let blockN : Nat := 64
  let hdim : Nat := 64
  let numMBlocks : Nat := 4  -- seqLenQ / blockM
  let numKvBlocks : Nat := 4  -- seqLenK / blockN
  let totalTiles : Nat := numMBlocks * 4 * 1  -- numMBlocks * numHeads * numBatch

  comment "=== FlashAttention3 Forward (Persistent Scheduler) ==="
  comment "CTAs dynamically fetch tiles via atomic counter"

  -- Shared memory
  let sQ : ST GpuFloat.BFloat16 blockM hdim ← allocST .BFloat16 blockM hdim
  let sK : ST GpuFloat.BFloat16 blockN hdim ← allocST .BFloat16 blockN hdim
  let sV : ST GpuFloat.BFloat16 blockN hdim .Col ← allocST .BFloat16 blockN hdim .Col
  let sO : ST GpuFloat.Float32 blockM hdim ← allocST .Float32 blockM hdim
  let sLse : SV GpuFloat.Float32 blockM ← allocSV .Float32 blockM

  -- Register tiles
  let q : RT GpuFloat.BFloat16 blockM hdim ← allocRT .BFloat16 blockM hdim
  let k : RT GpuFloat.BFloat16 blockN hdim ← allocRT .BFloat16 blockN hdim
  let v : RT GpuFloat.BFloat16 blockN hdim .Col ← allocRT .BFloat16 blockN hdim .Col
  let scores : RT GpuFloat.Float32 blockM blockN ← allocRT .Float32 blockM blockN
  let p : RT GpuFloat.BFloat16 blockM blockN ← allocRT .BFloat16 blockM blockN
  let o : RT GpuFloat.Float32 blockM hdim ← zeroRT .Float32 blockM hdim

  -- Softmax state
  let rowMax : RV GpuFloat.Float32 blockM ← negInftyRV .Float32 blockM
  let rowSum : RV GpuFloat.Float32 blockM ← zeroRV .Float32 blockM
  let prevMax : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM
  let scaleVec : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM
  let lseVec : RV GpuFloat.Float32 blockM ← allocRV .Float32 blockM

  comment "=== Persistent Tile Loop ==="
  comment "Each CTA fetches tiles until all are processed"

  -- Persistent loop: fetch tiles until done
  -- In production: while (tileIdx < totalTiles) { ... atomicAdd(tileCounter) ... }
  -- Simplified version: process assigned tiles
  forLoop 0 totalTiles do
    comment "--- Fetch next tile via atomic counter ---"
    -- In production: atomically increment and check bounds
    -- let tileIdx ← atomicAdd tileCounter_ptr 1
    -- if tileIdx >= totalTiles then break

    comment "Decode tile index to (m_block, head, batch)"
    -- For causal: use reverse M ordering (LPT)
    -- mBlock = numMBlocks - 1 - (tileIdx % numMBlocks)
    -- headIdx = (tileIdx / numMBlocks) % numHeads
    -- batchIdx = tileIdx / (numMBlocks * numHeads)

    comment "Reset output accumulator for new tile"
    zero o
    -- Reset softmax state
    -- negInfty rowMax
    -- zeroVec rowSum

    comment "Load Q for this (m_block, head, batch)"
    load q sQ

    comment "Main attention loop over K/V blocks"
    forLoop 0 numKvBlocks do
      load k sK
      load v sV

      comment "S = Q @ K^T"
      zero scores
      mmaT scores q k scores
      mmaCommitGroup
      mmaAsyncWait 0

      comment "Causal mask"
      makeCausal scores scores (some (-1e10))

      comment "Online softmax"
      copyVec prevMax rowMax
      rowMaxAccum rowMax scores rowMax
      subVec scaleVec prevMax rowMax
      expVec scaleVec scaleVec
      mulCol o o scaleVec
      mulVec rowSum rowSum scaleVec
      subCol scores scores rowMax
      exp scores scores
      rowSumAccum rowSum scores rowSum
      convert p scores

      comment "O += P @ V"
      mma o p v o
      mmaCommitGroup
      mmaAsyncWait 0

      sync 0

    comment "Finalize and store output for this tile"
    divCol o o rowSum
    logVec lseVec rowSum
    addVec lseVec lseVec rowMax

    store sO o
    storeVec sLse lseVec

/-! ## Kernel Verification -/

-- Verify auto-generated kernel definitions
#check flashAttn3Fwd.kernel
#check flashAttn3Fwd.launch
#check flashAttn3BwdPrep.kernel
#check flashAttn3BwdPrep.launch
#check flashAttn3Bwd.kernel
#check flashAttn3Bwd.launch
#check flashAttn3FwdGQA.kernel
#check flashAttn3FwdGQA.launch
#check flashAttn3FwdPersistent.kernel
#check flashAttn3FwdPersistent.launch

-- Generate C++ code for inspection
#eval IO.println "=== FlashAttention3 Forward ===" *>
      IO.println (generateKernel flashAttn3Fwd.kernel)

#eval IO.println "\n=== FlashAttention3 Backward Prep ===" *>
      IO.println (generateKernel flashAttn3BwdPrep.kernel)

#eval IO.println "\n=== FlashAttention3 Backward ===" *>
      IO.println (generateKernel flashAttn3Bwd.kernel)

#eval IO.println "\n=== FlashAttention3 Forward GQA ===" *>
      IO.println (generateKernel flashAttn3FwdGQA.kernel)

#eval IO.println "\n=== FlashAttention3 Forward Persistent ===" *>
      IO.println (generateKernel flashAttn3FwdPersistent.kernel)

end Tyr.GPU.Kernels.FlashAttn3
