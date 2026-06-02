/-
  Tyr/GPU/Codegen/Pipeline.lean

  N-stage ring buffers and structured pipelined loop emission.
  Generalizes DoubleBuffer (Macros.lean) to arbitrary pipeline depths
  and provides prologue/main/epilogue loop splitting with optional
  warp specialization.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.Macros

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Ring Buffer

An N-stage ring buffer holding `numStages` shared tiles and semaphores.
Stage selection uses modular arithmetic on the iteration index.
-/

/-- N-stage ring buffer for pipelined shared memory access -/
structure RingBuffer (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout) (numStages : Nat) where
  /-- Shared tiles for each pipeline stage -/
  tiles : Array (ST dtype rows cols layout)
  /-- Semaphores for each pipeline stage -/
  sems : Array Semaphore
  deriving Repr

/-- Allocate a ring buffer with `numStages` shared tiles and semaphores.
    Semaphore 0 is initialized to 1 (ready for first write), others to 0. -/
def allocRingBuffer (dtype : GpuFloat) (rows cols : Nat)
    (layout : TileLayout := .Row) (numStages : Nat := 2)
    : KernelM (RingBuffer dtype rows cols layout numStages) := do
  let mut tiles : Array (ST dtype rows cols layout) := #[]
  let mut sems : Array Semaphore := #[]
  for _ in List.range numStages do
    let tile ← allocST dtype rows cols layout
    let sem ← allocSemaphore
    tiles := tiles.push tile
    sems := sems.push sem
  -- Initialize: first semaphore ready for write, others not
  for i in List.range numStages do
    match sems[i]? with
    | some sem => initSemaphore sem (if i == 0 then 1 else 0)
    | none => pure ()
  pure { tiles, sems }

/-- Get the shared tile at a given compile-time stage index (modulo numStages) -/
def RingBuffer.atStage {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    {numStages : Nat}
    (rb : RingBuffer dtype rows cols layout numStages)
    (stage : Nat) : ST dtype rows cols layout :=
  match rb.tiles[stage % numStages]? with
  | some tile => tile
  | none => ⟨⟨0⟩⟩  -- unreachable if numStages > 0

/-- Get the semaphore at a given compile-time stage index -/
def RingBuffer.semAtStage {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    {numStages : Nat}
    (rb : RingBuffer dtype rows cols layout numStages)
    (stage : Nat) : Semaphore :=
  match rb.sems[stage % numStages]? with
  | some sem => sem
  | none => ⟨⟨0⟩⟩  -- unreachable if numStages > 0

/-- Number of shared tiles in the ring buffer -/
def RingBuffer.size {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    {numStages : Nat}
    (_ : RingBuffer dtype rows cols layout numStages) : Nat :=
  numStages

/-! ## Pipeline Configuration -/

/-- Configuration for a pipelined loop -/
structure PipelineConfig where
  /-- Total number of iterations -/
  numIters : Nat
  /-- Pipeline depth (number of stages to prefetch) -/
  depth : Nat := 2
  /-- Use warp specialization (producer wg0, consumer wg1) -/
  warpSpecialized : Bool := false
  /-- Use ShapeSync-derived semaphore handoff for depth-1 warp-specialized pipelines. -/
  useShapeSyncBarriers : Bool := false
  deriving Repr, Inhabited

/-! ## Pipelined Ring Loop

Emits a structured loop with:
- **Prologue**: fills `depth` pipeline stages (producer only)
- **Main loop**: producer fills stage `i+depth` while consumer processes stage `i`
- **Epilogue**: drains remaining `depth` stages (consumer only)

When `warpSpecialized = true`, producer/consumer run in separate warp groups.
-/

/-- Emit a pipelined ring loop with producer/consumer callbacks.

    `producer stage` is called with the pipeline stage index to fill.
    `consumer stage` is called with the pipeline stage index to consume.

    The loop structure is:
    - Prologue: producer fills stages 0..depth-1
    - Main: for i in 0..numIters-depth: producer(i+depth) || consumer(i)
    - Epilogue: consumer drains stages numIters-depth..numIters-1 -/
def pipelinedRingLoop (cfg : PipelineConfig)
    (producer : Nat → KernelM Unit)
    (consumer : Nat → KernelM Unit) : KernelM Unit := do
  let depth := Nat.min cfg.depth cfg.numIters
  let mut shapeSyncProtocol? : Option Tyr.ShapeSync.BarrierProtocol := none
  if cfg.warpSpecialized && depth > 0 then
    let proofCtx := (← get).proof.threadCtx
    for stage in List.range depth do
      let representativeProducer ← captureBodySnapshot (producer stage)
      let representativeConsumer ← captureBodySnapshot (consumer stage)
      let previewStmts : Array KStmt := #[
        .ifWarpGroup 0 representativeProducer,
        .ifWarpGroup 1 representativeConsumer
      ]
      match KernelProof.shapeSyncAnalysisFromStmts proofCtx previewStmts with
      | some analysis =>
          addShapeSyncPipelineAnalysis analysis
          match shapeSyncProtocol?, analysis.protocols.head? with
          | none, some protocol => shapeSyncProtocol? := some protocol
          | _, _ => pure ()
      | none => pure ()
  let useShapeSyncHandoff :=
    cfg.warpSpecialized && cfg.useShapeSyncBarriers && depth == 1 &&
      shapeSyncProtocol?.isSome
  let forwardSem? ←
    if useShapeSyncHandoff then
      let sem ← allocSemaphore
      initSemaphore sem 0
      pure (some sem)
    else
      pure none
  let backpressureSem? ←
    if useShapeSyncHandoff then
      let sem ← allocSemaphore
      initSemaphore sem 1
      pure (some sem)
    else
      pure none
  let runProducer (action : KernelM Unit) : KernelM Unit := do
    match shapeSyncProtocol?, forwardSem?, backpressureSem? with
    | some protocol, some forwardSem, some backpressureSem =>
        waitSemaphore backpressureSem
        action
        arriveSemaphore forwardSem protocol.forward.expectedParticipants
    | _, _, _ =>
        action
  let runConsumer (action : KernelM Unit) : KernelM Unit := do
    match shapeSyncProtocol?, forwardSem?, backpressureSem? with
    | some protocol, some forwardSem, some backpressureSem =>
        waitSemaphore forwardSem
        action
        arriveSemaphore backpressureSem protocol.backpressure.expectedParticipants
    | _, _, _ =>
        action
  let wrap := if cfg.warpSpecialized then
    fun (wg : Nat) (action : KernelM Unit) => ifWarpGroup wg action
  else
    fun _ (action : KernelM Unit) => action

  -- Prologue: fill pipeline stages
  comment s!"Pipeline prologue: fill {depth} stages"
  for i in List.range depth do
    wrap 0 (if useShapeSyncHandoff then runProducer (producer i) else producer i)

  if !useShapeSyncHandoff then
    sync

  -- Main loop: overlap producer and consumer
  if cfg.numIters > depth then
    comment s!"Pipeline main loop: {cfg.numIters - depth} iterations"
    forLoop 0 (cfg.numIters - depth) do
      -- Producer fills next stage (using representative stage index)
      wrap 0 (if useShapeSyncHandoff then runProducer (producer depth) else producer depth)
      -- Consumer processes current stage
      wrap 1 (if useShapeSyncHandoff then runConsumer (consumer 0) else consumer 0)
      if !useShapeSyncHandoff then
        sync

  -- Epilogue: drain pipeline
  comment s!"Pipeline epilogue: drain {depth} stages"
  for i in List.range depth do
    wrap 1 (if useShapeSyncHandoff then runConsumer (consumer i) else consumer i)
    if !useShapeSyncHandoff then
      sync

end Tyr.GPU.Codegen
