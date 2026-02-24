/-
  Tyr/GPU/Tile.lean

  Tile typeclasses and concrete tile types mapping to ThunderKittens
  rt<T, rows, cols, layout>, st<T, rows, cols, layout>, etc.
-/
import Tyr.GPU.Types
import Tyr.GPU.Capabilities

/-!
# `Tyr.GPU.Tile`

Defines tile typeclasses and concrete register and shared-memory tile structures used by GPU codegen.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU

/-- Base tile concept - all tiles implement this -/
class Tile (α : Type) where
  dtype : GpuFloat
  rows : Nat
  cols : Nat
  location : TileLoc
  layout : TileLayout

/-- Register tiles (thread-local) -/
class RegisterTile (α : Type) extends Tile α where
  location_is_register : location = TileLoc.Register
  /-- Number of 16×16 base tiles vertically -/
  height : Nat := rows / 16
  /-- Number of 16×16 base tiles horizontally -/
  width : Nat := cols / 16

/-- Shared tiles (warp-cooperative with swizzling) -/
class SharedTile (α : Type) extends Tile α where
  location_is_shared : location = TileLoc.Shared
  /-- Swizzle pattern for bank conflict avoidance -/
  swizzleMode : SwizzleMode := .Swizzle128B

/-- Register tile: rt<T, rows, cols, layout> -/
structure RT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  /-- Placeholder - in codegen this becomes a declaration -/
  mk ::
  deriving Repr, Inhabited

instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    Tile (RT dtype rows cols layout) where
  dtype := dtype
  rows := rows
  cols := cols
  location := .Register
  layout := layout

instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    RegisterTile (RT dtype rows cols layout) where
  location_is_register := rfl
  height := rows / 16
  width := cols / 16

/-- Shared tile: st<T, rows, cols, layout> -/
structure ST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  mk ::
  deriving Repr, Inhabited

instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    Tile (ST dtype rows cols layout) where
  dtype := dtype
  rows := rows
  cols := cols
  location := .Shared
  layout := layout

instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    SharedTile (ST dtype rows cols layout) where
  location_is_shared := rfl

/-- Register vector: rv<T, length> -/
structure RV (dtype : GpuFloat) (length : Nat) where
  mk ::
  deriving Repr, Inhabited

/-- Shared vector: sv<T, length> -/
structure SV (dtype : GpuFloat) (length : Nat) where
  mk ::
  deriving Repr, Inhabited

/-- Global layout for device memory -/
structure GL (dtype : GpuFloat) where
  shape : Array Nat
  strides : Array Nat
  deriving Repr, Inhabited

/-- Compute total elements in a tile -/
def Tile.elements [t : Tile α] : Nat :=
  t.rows * t.cols

/-- Compute bytes for a tile -/
def Tile.bytes [t : Tile α] : Nat :=
  t.rows * t.cols * t.dtype.bytes

/-- Check if tile dimensions are valid (multiples of 16 for MMA) -/
def Tile.validForMMA [t : Tile α] : Bool :=
  t.rows % 16 = 0 && t.cols % 16 = 0

end Tyr.GPU
