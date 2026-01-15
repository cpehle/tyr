/-
  Tyr/GPU/Codegen/GlobalLayout.lean

  Global memory layout abstractions following ThunderKittens patterns.
  Provides GlobalLayout (gl) and TileCoord types for memory access.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Tile Coordinate

Tile-level coordinates for global memory access.
Mirrors ThunderKittens' coord<T> type with batch, depth, row, col dimensions.
-/

/-- Tile-level coordinate for global memory access.
    All dimensions default to 0 for simple 2D cases. -/
structure TileCoord where
  /-- Batch dimension index -/
  b : Nat := 0
  /-- Depth dimension index (e.g., attention head) -/
  d : Nat := 0
  /-- Row tile index -/
  r : Nat := 0
  /-- Column tile index -/
  c : Nat := 0
  deriving Repr, Inhabited, BEq

/-- Create a row-only coordinate -/
def TileCoord.row (r : Nat) : TileCoord := { r := r }

/-- Create a row-column coordinate -/
def TileCoord.rowCol (r c : Nat) : TileCoord := { r := r, c := c }

/-- Create a batch-row coordinate -/
def TileCoord.batchRow (b r : Nat) : TileCoord := { b := b, r := r }

/-- Create a depth-row coordinate (for attention heads) -/
def TileCoord.headRow (d r : Nat) : TileCoord := { d := d, r := r }

/-! ## Runtime Tile Coordinate

For coordinates computed at runtime (from blockIdx, loop indices, etc.)
-/

/-- Runtime tile coordinate with VarIds for dynamic indices -/
structure RTileCoord where
  /-- Batch dimension (VarId for runtime value) -/
  b : VarId
  /-- Depth dimension -/
  d : VarId
  /-- Row tile index -/
  r : VarId
  /-- Column tile index -/
  c : VarId
  deriving Repr, Inhabited

/-! ## Global Memory Layout

4D global memory layout following ThunderKittens' gl<T, b, d, r, c> pattern.
Bundles a GPtr with shape information for type-safe memory operations.
-/

/-- Global memory layout with 4D shape (batch, depth, rows, cols).
    Mirrors ThunderKittens' gl<T, b, d, r, c> type.

    Shape dimensions are element counts, not tile counts.
    For example, a [32, 8, 1024, 64] tensor for batch=32, heads=8, seq=1024, dim=64. -/
structure GlobalLayout (dtype : GpuFloat) (batch depth rows cols : Nat) where
  /-- Underlying global memory pointer -/
  ptr : GPtr dtype
  deriving Repr

/-- Short alias for GlobalLayout -/
abbrev GL := GlobalLayout

/-- 2D global layout (rows × cols), batch=1, depth=1 -/
abbrev GL2 (dtype : GpuFloat) (rows cols : Nat) := GlobalLayout dtype 1 1 rows cols

/-- 3D global layout (depth × rows × cols), batch=1 -/
abbrev GL3 (dtype : GpuFloat) (depth rows cols : Nat) := GlobalLayout dtype 1 depth rows cols

/-- 4D global layout (batch × depth × rows × cols) -/
abbrev GL4 := GlobalLayout

/-! ## GlobalLayout Accessors -/

/-- Get the underlying GPtr -/
def GlobalLayout.toGPtr {dtype : GpuFloat} {b d r c : Nat}
    (gl : GlobalLayout dtype b d r c) : GPtr dtype := gl.ptr

/-- Get the VarId of the underlying pointer -/
def GlobalLayout.varId {dtype : GpuFloat} {b d r c : Nat}
    (gl : GlobalLayout dtype b d r c) : VarId := gl.ptr.id

/-- Get batch size -/
def GlobalLayout.batchSize {dtype : GpuFloat} {b d r c : Nat}
    (_ : GlobalLayout dtype b d r c) : Nat := b

/-- Get depth (number of heads) -/
def GlobalLayout.depthSize {dtype : GpuFloat} {b d r c : Nat}
    (_ : GlobalLayout dtype b d r c) : Nat := d

/-- Get number of rows -/
def GlobalLayout.numRows {dtype : GpuFloat} {b d r c : Nat}
    (_ : GlobalLayout dtype b d r c) : Nat := r

/-- Get number of columns -/
def GlobalLayout.numCols {dtype : GpuFloat} {b d r c : Nat}
    (_ : GlobalLayout dtype b d r c) : Nat := c

/-! ## Stride Computation

Following ThunderKittens' row-major layout convention:
stride<0> = depth * rows * cols  (batch stride)
stride<1> = rows * cols          (depth stride)
stride<2> = cols                 (row stride)
stride<3> = 1                    (col stride)
-/

/-- Compute batch stride (elements between batches) -/
def GlobalLayout.batchStride {dtype : GpuFloat} {b d r c : Nat}
    (_ : GlobalLayout dtype b d r c) : Nat := d * r * c

/-- Compute depth stride (elements between depths/heads) -/
def GlobalLayout.depthStride {dtype : GpuFloat} {b d r c : Nat}
    (_ : GlobalLayout dtype b d r c) : Nat := r * c

/-- Compute row stride (elements between rows) -/
def GlobalLayout.rowStride {dtype : GpuFloat} {b d r c : Nat}
    (_ : GlobalLayout dtype b d r c) : Nat := c

/-- Total number of elements -/
def GlobalLayout.numel {dtype : GpuFloat} {b d r c : Nat}
    (_ : GlobalLayout dtype b d r c) : Nat := b * d * r * c

/-- Compute linear element offset from a static tile coordinate.
    Note: This computes element offset, not tile offset.
    For tile-based access, multiply coord.r by tileRows, etc. -/
def GlobalLayout.elementOffset {dtype : GpuFloat} {b d r c : Nat}
    (gl : GlobalLayout dtype b d r c) (coord : TileCoord) : Nat :=
  coord.b * gl.batchStride + coord.d * gl.depthStride +
  coord.r * gl.rowStride + coord.c

/-- Compute tile offset given tile dimensions -/
def GlobalLayout.tileOffset {dtype : GpuFloat} {b d r c : Nat}
    (gl : GlobalLayout dtype b d r c) (coord : TileCoord)
    (tileRows tileCols : Nat) : Nat :=
  coord.b * gl.batchStride + coord.d * gl.depthStride +
  coord.r * tileRows * gl.rowStride + coord.c * tileCols

/-! ## Block/Thread Index Helpers

Functions to get thread block and thread indices at runtime.
-/

/-- Get blockIdx.x as a runtime value -/
def getBlockIdxX : KernelM VarId := do
  let v ← freshVar
  emit (.getBlockIdx v 0)
  pure v

/-- Get blockIdx.y as a runtime value -/
def getBlockIdxY : KernelM VarId := do
  let v ← freshVar
  emit (.getBlockIdx v 1)
  pure v

/-- Get blockIdx.z as a runtime value -/
def getBlockIdxZ : KernelM VarId := do
  let v ← freshVar
  emit (.getBlockIdx v 2)
  pure v

/-- Get threadIdx.x as a runtime value -/
def getThreadIdxX : KernelM VarId := do
  let v ← freshVar
  emit (.getThreadIdx v 0)
  pure v

/-- Get threadIdx.y as a runtime value -/
def getThreadIdxY : KernelM VarId := do
  let v ← freshVar
  emit (.getThreadIdx v 1)
  pure v

/-- Get threadIdx.z as a runtime value -/
def getThreadIdxZ : KernelM VarId := do
  let v ← freshVar
  emit (.getThreadIdx v 2)
  pure v

/-- Create a runtime coordinate with batch=0, depth=0, row=blockIdx.y, col=blockIdx.x
    This is the common pattern for 2D tiled kernels. -/
def blockCoord2D : KernelM RTileCoord := do
  let batchId ← freshVar
  emit (.constInt batchId 0)
  let depthId ← freshVar
  emit (.constInt depthId 0)
  let rowId ← getBlockIdxY
  let colId ← getBlockIdxX
  pure { b := batchId, d := depthId, r := rowId, c := colId }

/-- Create a runtime coordinate from explicit VarIds -/
def makeRTileCoord (b d r c : VarId) : RTileCoord :=
  { b := b, d := d, r := r, c := c }

/-- Create a runtime coordinate with only row specified (batch=0, depth=0, col=0) -/
def rowCoord (r : VarId) : KernelM RTileCoord := do
  let zero ← freshVar
  emit (.constInt zero 0)
  pure { b := zero, d := zero, r := r, c := zero }

/-- Create a runtime coordinate with row from loop index -/
def loopRowCoord (loopVar : VarId) : KernelM RTileCoord := do
  let zero ← freshVar
  emit (.constInt zero 0)
  pure { b := zero, d := zero, r := loopVar, c := zero }

/-! ## Typeclass for Global Layout

Enables generic operations over GlobalLayout types.
-/

/-- Typeclass for global layout types -/
class IsGlobalLayout (T : Type) where
  dtype : GpuFloat
  batch : Nat
  depth : Nat
  rows : Nat
  cols : Nat

instance {dtype : GpuFloat} {b d r c : Nat} :
    IsGlobalLayout (GlobalLayout dtype b d r c) where
  dtype := dtype
  batch := b
  depth := d
  rows := r
  cols := c

/-! ## Helper Functions for Common Patterns -/

/-- Number of row tiles given tile size -/
def GlobalLayout.numRowTiles {dtype : GpuFloat} {b d r c : Nat}
    (gl : GlobalLayout dtype b d r c) (tileRows : Nat) : Nat :=
  (gl.numRows + tileRows - 1) / tileRows

/-- Number of column tiles given tile size -/
def GlobalLayout.numColTiles {dtype : GpuFloat} {b d r c : Nat}
    (gl : GlobalLayout dtype b d r c) (tileCols : Nat) : Nat :=
  (gl.numCols + tileCols - 1) / tileCols

/-! ## Global Memory Operations with RTileCoord

These functions provide a cleaner API for loading/storing data
from global memory using RTileCoord structs.
-/

/-- Load tile from global memory to shared memory using RTileCoord.
    Usage: `loadGlobal dst src coord` -/
def loadGlobal {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coord : RTileCoord)
    : KernelM Unit := do
  emit (.loadGlobal dst.id src.id coord.b coord.d coord.r coord.c)

/-- Store tile from shared memory to global memory using RTileCoord.
    Usage: `storeGlobal dst src coord` -/
def storeGlobal {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coord : RTileCoord)
    : KernelM Unit := do
  emit (.storeGlobal dst.id src.id coord.b coord.d coord.r coord.c)

/-- Async load from global to shared with semaphore (TMA).
    Usage: `loadGlobalAsync dst src coord sem` -/
def loadGlobalAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coord : RTileCoord)
    (sem : VarId)  -- Semaphore VarId
    : KernelM Unit := do
  emit (.loadGlobalAsync dst.id src.id coord.b coord.d coord.r coord.c sem)

/-- Async store from shared to global (TMA).
    Usage: `storeGlobalAsync dst src coord` -/
def storeGlobalAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coord : RTileCoord)
    : KernelM Unit := do
  emit (.storeGlobalAsync dst.id src.id coord.b coord.d coord.r coord.c)

/-- Atomic add store from shared to global (for gradient accumulation).
    Usage: `storeGlobalAdd dst src coord` -/
def storeGlobalAdd {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coord : RTileCoord)
    : KernelM Unit := do
  emit (.storeGlobalAdd dst.id src.id coord.b coord.d coord.r coord.c)

/-- Create an RTileCoord with a modified row index (for loop iteration).
    Common pattern: `coord.withRow loopIdx.id` -/
def RTileCoord.withRow (coord : RTileCoord) (newR : VarId) : RTileCoord :=
  { coord with r := newR }

/-- Create an RTileCoord with a modified column index.
    Common pattern: `coord.withCol loopIdx.id` -/
def RTileCoord.withCol (coord : RTileCoord) (newC : VarId) : RTileCoord :=
  { coord with c := newC }

end Tyr.GPU.Codegen
