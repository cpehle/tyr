/-
  Tyr/GPU/Codegen/TileTypes.lean

  Dependent types for GPU tiles with dimensions embedded in types.
  Enables compile-time checking of dimension compatibility for operations.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- Register tile with dimensions embedded in type -/
structure RT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  id : VarId
  deriving Repr

/-- Shared memory tile -/
structure ST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  id : VarId
  deriving Repr

/-- Register vector -/
structure RV (dtype : GpuFloat) (len : Nat) where
  id : VarId
  deriving Repr

/-- Shared vector -/
structure SV (dtype : GpuFloat) (len : Nat) where
  id : VarId
  deriving Repr

/-- GPU pointer (for kernel parameters) -/
structure Ptr (dtype : GpuFloat) where
  name : String
  deriving Repr

/-- Typeclass for tile types -/
class IsTile (T : Type) where
  dtype : GpuFloat
  rows : Nat
  cols : Nat
  layout : TileLayout
  location : TileLoc

/-- Typeclass for vector types -/
class IsVec (T : Type) where
  dtype : GpuFloat
  len : Nat
  location : TileLoc

-- Instances for RT
instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    IsTile (RT dtype rows cols layout) where
  dtype := dtype
  rows := rows
  cols := cols
  layout := layout
  location := .Register

-- Instances for ST
instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    IsTile (ST dtype rows cols layout) where
  dtype := dtype
  rows := rows
  cols := cols
  layout := layout
  location := .Shared

-- Instances for RV
instance {dtype : GpuFloat} {len : Nat} : IsVec (RV dtype len) where
  dtype := dtype
  len := len
  location := .Register

-- Instances for SV
instance {dtype : GpuFloat} {len : Nat} : IsVec (SV dtype len) where
  dtype := dtype
  len := len
  location := .Shared

/-- Get VarId from register tile -/
def RT.varId {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : RT dtype rows cols layout) : VarId := t.id

/-- Get VarId from shared tile -/
def ST.varId {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : ST dtype rows cols layout) : VarId := t.id

/-- Get VarId from register vector -/
def RV.varId {dtype : GpuFloat} {len : Nat} (v : RV dtype len) : VarId := v.id

/-- Get VarId from shared vector -/
def SV.varId {dtype : GpuFloat} {len : Nat} (v : SV dtype len) : VarId := v.id

end Tyr.GPU.Codegen
