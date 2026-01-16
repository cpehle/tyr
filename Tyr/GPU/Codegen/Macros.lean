/-
  Tyr/GPU/Codegen/Macros.lean

  High-level pattern macros for common GPU kernel patterns.
  These expand to sequences of lower-level operations.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Ops

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Double Buffering Helper

Structure and functions for managing double-buffered pipeline stages.
-/

/-- Double buffer state for a single tile -/
structure DoubleBuffer (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout) where
  /-- First buffer -/
  buf0 : ST dtype rows cols layout
  /-- Second buffer -/
  buf1 : ST dtype rows cols layout
  /-- Semaphore for buf0 -/
  sem0 : Semaphore
  /-- Semaphore for buf1 -/
  sem1 : Semaphore
  deriving Repr

/-- Allocate a double buffer with semaphores -/
def allocDoubleBuffer (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelM (DoubleBuffer dtype rows cols layout) := do
  let buf0 ← allocST dtype rows cols layout
  let buf1 ← allocST dtype rows cols layout
  let sem0 ← allocSemaphore
  let sem1 ← allocSemaphore
  -- Initialize semaphores: buf0 ready for write (1), buf1 not ready (0)
  initSemaphore sem0 1
  initSemaphore sem1 0
  pure { buf0, buf1, sem0, sem1 }

/-- Get the current buffer based on iteration parity -/
def DoubleBuffer.current {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (db : DoubleBuffer dtype rows cols layout)
    (iteration : Nat) : ST dtype rows cols layout :=
  if iteration % 2 == 0 then db.buf0 else db.buf1

/-- Get the current semaphore based on iteration parity -/
def DoubleBuffer.currentSem {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (db : DoubleBuffer dtype rows cols layout)
    (iteration : Nat) : Semaphore :=
  if iteration % 2 == 0 then db.sem0 else db.sem1

/-- Get the alternate semaphore based on iteration parity -/
def DoubleBuffer.altSem {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (db : DoubleBuffer dtype rows cols layout)
    (iteration : Nat) : Semaphore :=
  if iteration % 2 == 0 then db.sem1 else db.sem0

/-! ## Warp Specialization Helpers

Functions for producer/consumer warp group patterns.
-/

/-- Execute code only in producer warp group (warpgroup 0) -/
def asProducer (action : KernelM Unit) : KernelM Unit :=
  ifWarpGroup 0 action

/-- Execute code only in consumer warp group (warpgroup 1) -/
def asConsumer (action : KernelM Unit) : KernelM Unit :=
  ifWarpGroup 1 action

/-- Execute code in a specific warp group -/
def inWarpGroup (wgIdx : Nat) (action : KernelM Unit) : KernelM Unit :=
  ifWarpGroup wgIdx action

/-! ## Online Softmax Pattern

The online softmax pattern tracks running max and sum for numerical stability.
This is the core pattern used in FlashAttention.
-/

/-- Online softmax state -/
structure SoftmaxState (dtype : GpuFloat) (rows : Nat) where
  /-- Running row-wise maximum -/
  rowMax : RV dtype rows
  /-- Running row-wise sum of exp(x - max) -/
  rowSum : RV dtype rows
  /-- Previous maximum (for rescaling) -/
  prevMax : RV dtype rows
  /-- Scale factor exp(prevMax - newMax) -/
  scale : RV dtype rows
  deriving Repr

/-- Allocate online softmax state -/
def allocSoftmaxState (dtype : GpuFloat) (rows : Nat)
    : KernelM (SoftmaxState dtype rows) := do
  let rowMax ← negInftyRV dtype rows
  let rowSum ← zeroRV dtype rows
  let prevMax ← allocRV dtype rows
  let scale ← allocRV dtype rows
  pure { rowMax, rowSum, prevMax, scale }

/-- Apply online softmax update to scores tile, rescaling output accumulator.
    This is the core attention pattern:
    1. Update running max
    2. Compute scale = exp(prevMax - newMax)
    3. Rescale output accumulator by scale
    4. Rescale running sum by scale
    5. Compute exp(scores - newMax)
    6. Update running sum

    Note: scores must have the same dtype as the accumulator (typically Float32)
    for numerical stability in the softmax computation. -/
def onlineSoftmax {dtype : GpuFloat} {rows cols outCols : Nat}
    (scores : RT dtype rows cols .Row)
    (output : RT dtype rows outCols .Row)
    (state : SoftmaxState dtype rows)
    : KernelM Unit := do
  -- 1. Save previous max
  copyVec state.prevMax state.rowMax

  -- 2. Update running max with new scores
  rowMaxAccum state.rowMax scores state.rowMax

  -- 3. Compute scale = exp(prevMax - newMax)
  subVec state.scale state.prevMax state.rowMax
  expVec state.scale state.scale

  -- 4. Rescale output accumulator: output *= scale (column broadcast)
  mulCol output output state.scale

  -- 5. Rescale running sum: rowSum *= scale
  mulVec state.rowSum state.rowSum state.scale

  -- 6. Compute exp(scores - max) in place
  subCol scores scores state.rowMax
  exp scores scores

  -- 7. Update running sum
  rowSumAccum state.rowSum scores state.rowSum

/-- Finalize softmax: divide output by row sum -/
def finalizeSoftmax {accDtype : GpuFloat} {rows cols : Nat}
    (output : RT accDtype rows cols .Row)
    (state : SoftmaxState accDtype rows)
    : KernelM Unit := do
  divCol output output state.rowSum

/-- Compute log-sum-exp from softmax state: lse = log(rowSum) + rowMax -/
def computeLSE {dtype : GpuFloat} {rows : Nat}
    (state : SoftmaxState dtype rows)
    : KernelM (RV dtype rows) := do
  let lse ← allocRV dtype rows
  logVec lse state.rowSum
  addVec lse lse state.rowMax
  pure lse

/-! ## Pipeline Iteration Pattern

Helper for pipelined loop iterations with double buffering.
-/

/-- State for a pipelined loop iteration -/
structure PipelineIter where
  /-- Current iteration index -/
  iter : Nat
  /-- Total number of iterations -/
  total : Nat
  /-- Whether this is the first iteration -/
  isFirst : Bool
  /-- Whether this is the last iteration -/
  isLast : Bool
  deriving Repr

/-- Create pipeline iteration info -/
def mkPipelineIter (iter total : Nat) : PipelineIter :=
  { iter, total, isFirst := iter == 0, isLast := iter == total - 1 }

/-! ## Common Attention Patterns

Pre-packaged patterns for attention computation.
-/

/-- Execute a single attention block iteration.
    Computes: S = Q @ K^T, P = softmax(S), O += P @ V

    Dimensions:
    - Q: [blockM, headDim] (query tile)
    - K: [blockN, headDim] (key tile, will be transposed)
    - V: [blockN, headDim] (value tile, col-major for MMA)
    - S = Q @ K^T: [blockM, blockN] (attention scores)
    - P = softmax(S): [blockM, blockN] (attention probabilities)
    - O += P @ V: [blockM, headDim] (output accumulator)
-/
def attentionBlockIter {blockM blockN headDim : Nat} {inDtype accDtype : GpuFloat}
    (q : RT inDtype blockM headDim .Row)       -- Query tile [blockM, headDim]
    (k : RT inDtype blockN headDim .Row)       -- Key tile [blockN, headDim] (transposed in mmaT)
    (v : RT inDtype blockN headDim .Col)       -- Value tile [blockN, headDim] col-major
    (output : RT accDtype blockM headDim .Row) -- Output accumulator [blockM, headDim]
    (state : SoftmaxState accDtype blockM)
    (isCausal : Bool := false)
    (maskVal : Float := -1e10)
    : KernelM Unit := do
  comment "Compute S = Q @ K^T"
  let scores ← allocRT accDtype blockM blockN .Row
  let zeros ← zeroRT accDtype blockM blockN .Row
  mmaT scores q k zeros

  -- Apply causal mask if needed
  if isCausal then
    comment "Apply causal mask"
    makeCausal scores scores (some maskVal)

  comment "Online softmax"
  onlineSoftmax scores output state

  comment "Convert scores to input dtype for P @ V"
  let p ← allocRT inDtype blockM blockN .Row
  convert p scores

  comment "Accumulate O += P @ V"
  mma output p v output

/-! ## Helper Macros via Syntax Extensions

These provide syntactic sugar for common patterns.
-/

-- Barrier naming helper
/-- Named barrier identifier -/
structure NamedBarrier where
  id : Nat
  numThreads : Nat
  deriving Repr

/-- Signal a named barrier (producer side) -/
def signalBarrier (b : NamedBarrier) : KernelM Unit :=
  namedBarrierArrive b.id b.numThreads

/-- Wait on a named barrier (consumer side) -/
def waitBarrier (b : NamedBarrier) : KernelM Unit :=
  namedBarrierSync b.id b.numThreads

/-- Common barrier for query ready signal in FA3 -/
def queryReadyBarrier : NamedBarrier := { id := 0, numThreads := 256 }

/-- Common barrier for KV ready signal in FA3 -/
def kvReadyBarrier : NamedBarrier := { id := 1, numThreads := 256 }

/-! ## Kernel Configuration Patterns

Structures for parameterizing kernel configurations.
-/

/-- Flash Attention configuration -/
structure FlashAttnConfig where
  /-- Block size for queries (M dimension) -/
  blockM : Nat := 64
  /-- Block size for keys/values (N dimension) -/
  blockN : Nat := 64
  /-- Head dimension (K dimension) -/
  headDim : Nat := 64
  /-- Input data type -/
  dtype : GpuFloat := .BFloat16
  /-- Accumulator data type -/
  accDtype : GpuFloat := .Float32
  /-- Whether to use causal masking -/
  isCausal : Bool := false
  /-- Number of pipeline stages -/
  numStages : Nat := 2
  deriving Repr, Inhabited

/-- Layer normalization configuration -/
structure LayerNormConfig where
  /-- Hidden dimension -/
  hiddenDim : Nat := 768
  /-- Epsilon for numerical stability -/
  eps : Float := 1e-5
  /-- Data type -/
  dtype : GpuFloat := .BFloat16
  deriving Repr, Inhabited

/-- Grouped Query Attention configuration -/
structure GQAConfig extends FlashAttnConfig where
  /-- Number of query heads -/
  numQHeads : Nat := 32
  /-- Number of key/value heads -/
  numKVHeads : Nat := 8
  deriving Repr, Inhabited

/-! ## Template Functions

High-level templates that can be specialized with configurations.
-/

/-- Template for FlashAttention forward pass.
    Users can customize by overriding specific parts. -/
def flashAttnTemplate (cfg : FlashAttnConfig)
    (customInit : KernelM Unit := pure ())
    (customFinalize : KernelM Unit := pure ())
    : KernelM Unit := do
  comment s!"FlashAttention Forward: blockM={cfg.blockM}, blockN={cfg.blockN}, headDim={cfg.headDim}"

  -- Custom initialization hook
  customInit

  -- Allocate query tile (placeholder - actual kernel would use this in loop)
  let _q ← allocRT cfg.dtype cfg.blockM cfg.headDim .Row

  -- Allocate output accumulator
  let output ← zeroRT cfg.accDtype cfg.blockM cfg.headDim .Row

  -- Allocate softmax state
  let state ← allocSoftmaxState cfg.accDtype cfg.blockM

  comment "Main KV loop would go here (use _q, output, state)"
  -- (Loop body would be filled in by the user)

  -- Finalize softmax
  finalizeSoftmax output state

  -- Custom finalization hook
  customFinalize

  comment "Store output"

end Tyr.GPU.Codegen
