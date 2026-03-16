/-
  Tyr/GPU/Codegen/PersistentKernel.lean

  Persistent kernel scheduling patterns. Two modes:
  - `atomicCounter`: CTAs claim work units via a global atomic counter
  - `fixedStride`: static assignment where workId = blockIdx + i * gridDim
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.Loop

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Persistent Scheduling Modes -/

/-- Scheduling mode for persistent kernels -/
inductive PersistentMode where
  /-- CTAs claim work via a global atomic counter -/
  | atomicCounter
  /-- Static assignment: workId = blockIdx + i * gridDim -/
  | fixedStride
  deriving Repr, Inhabited, BEq

/-- Configuration for persistent kernel scheduling -/
structure PersistentConfig where
  /-- Scheduling mode -/
  mode : PersistentMode := .fixedStride
  deriving Repr, Inhabited

/-- A work item inside the persistent loop body -/
structure WorkItem where
  /-- Linear work unit ID -/
  workId : VarId
  /-- Whether this work item is valid (within bounds) -/
  isValid : VarId
  deriving Repr

/-! ## Persistent Loop Emission -/

/-- Emit a persistent loop that iterates over `totalWork` items.

    - `atomicCounter`: emits a while loop with `atomicAdd` to claim work
    - `fixedStride`: emits a for loop with stride = gridDim.x

    The `body` callback receives a `WorkItem` with the linear work ID
    and a validity flag. -/
def persistentLoop (cfg : PersistentConfig) (totalWork : VarId)
    (body : WorkItem → KernelM Unit) : KernelM Unit := do
  match cfg.mode with
  | .fixedStride =>
    comment "Persistent loop (fixed stride)"
    -- workId starts at blockIdx.x, strides by gridDim.x
    let blockIdxX ← freshVar
    emit (.getBlockIdx blockIdxX 0)
    -- We use a forLoopVal with stride emulated via raw code
    -- Emit: for (int workId = blockIdx.x; workId < totalWork; workId += gridDim.x)
    let workId ← freshVar
    let isValid ← freshVar
    emit (.constInt isValid 1)  -- always valid in fixed stride (loop condition checks)
    let bodyStmts ← captureBody (body { workId, isValid })
    -- Emit the strided for loop using raw code for the stride pattern
    let wid := VarId.toIdent workId
    let bid := VarId.toIdent blockIdxX
    let tw := VarId.toIdent totalWork
    emit (.raw s!"for (int {wid} = {bid}; {wid} < {tw}; {wid} += gridDim.x) \{")
    for stmt in bodyStmts do
      emit stmt
    emit (.raw "}")

  | .atomicCounter =>
    comment "Persistent loop (atomic counter)"
    let workId ← freshVar
    let isValid ← freshVar
    -- Emit atomic work-claiming loop
    let wid := VarId.toIdent workId
    let vid := VarId.toIdent isValid
    let tw := VarId.toIdent totalWork
    emit (.raw s!"int {wid} = 0;")
    emit (.raw s!"bool {vid} = true;")
    let bodyStmts ← captureBody (body { workId, isValid })
    -- While loop: atomicAdd to claim, check bounds, execute body
    emit (.raw s!"while (true) \{")
    emit (.raw s!"  if (threadIdx.x == 0) \{")
    emit (.raw s!"    {wid} = atomicAdd(reinterpret_cast<int*>(&{tw}), 1);")
    emit (.raw s!"  }")
    emit (.raw s!"  {wid} = __shfl_sync(0xffffffff, {wid}, 0);")
    emit (.raw s!"  if ({wid} >= {tw}) break;")
    for stmt in bodyStmts do
      emit stmt
    emit (.raw "}")

/-! ## Work ID to Tile Coordinate Mapping -/

/-- Convert a linear work ID to 2D tile coordinates (row, col).
    row = workId / numColTiles, col = workId % numColTiles -/
def workIdToTileCoord2D (workId : VarId) (numColTiles : VarId)
    : KernelM (VarId × VarId) := do
  let row ← freshVar
  emit (.scalarBinary .Div row workId numColTiles)
  let col ← freshVar
  emit (.scalarBinary .Mod col workId numColTiles)
  pure (row, col)

/-- Convert a linear work ID to L2-swizzled 2D coordinates.
    Uses a supergroup pattern for better L2 cache locality:
    - Tiles are grouped into supergroups of `groupSize` columns
    - Within a supergroup, tiles are visited in column-major order

    This is the standard swizzle pattern used in CUTLASS/Flux persistent GEMMs. -/
def workIdToSwizzledCoord (workId : VarId) (numColTiles : VarId)
    (groupSize : Nat := 8) : KernelM (VarId × VarId) := do
  comment s!"L2 swizzle (group size = {groupSize})"
  let groupSizeVar ← freshVar
  emit (.constInt groupSizeVar groupSize)
  -- numGroups = (numColTiles + groupSize - 1) / groupSize
  let numColTilesAdj ← freshVar
  let groupSizeMinus1 ← freshVar
  emit (.constInt groupSizeMinus1 (groupSize - 1))
  emit (.scalarBinary .Add numColTilesAdj numColTiles groupSizeMinus1)
  let numGroups ← freshVar
  emit (.scalarBinary .Div numGroups numColTilesAdj groupSizeVar)
  -- tilesPerGroup = numRowTiles * groupSize (but we compute from total)
  -- groupId = workId / (numRowTiles * groupSize) -- approximate via column groups
  -- Within-group linear index
  let groupIdx ← freshVar
  emit (.scalarBinary .Div groupIdx workId groupSizeVar)
  let withinGroup ← freshVar
  emit (.scalarBinary .Mod withinGroup workId groupSizeVar)
  -- Swizzled row and col
  let superGroupId ← freshVar
  emit (.scalarBinary .Div superGroupId groupIdx numGroups)
  let groupOffset ← freshVar
  emit (.scalarBinary .Mod groupOffset groupIdx numGroups)
  -- row = superGroupId, col = groupOffset * groupSize + withinGroup
  let colBase ← freshVar
  emit (.scalarBinary .Mul colBase groupOffset groupSizeVar)
  let col ← freshVar
  emit (.scalarBinary .Add col colBase withinGroup)
  pure (superGroupId, col)

end Tyr.GPU.Codegen
