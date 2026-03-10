import Tyr.GPU.Types
import Tyr.GPU.Capabilities

/-!
# Tyr.GPU.Tile

`Tyr.GPU.Tile` defines the abstract tile model used by the DSL.
It focuses on the shared concepts (`Tile`, `RegisterTile`, `SharedTile`) and
generic helper functions over those concepts.

Concrete codegen handles such as `RT`, `ST`, `RV`, `SV`, and `GL` live in the
codegen layer. Keeping those handle definitions there avoids maintaining two
parallel carrier hierarchies with the same names.
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
