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

/-! ## Kernel Parameter Types

These types represent kernel parameters as first-class Lean values.
They wrap VarId so they can be used in operations like tmaLoad.
-/

/-- Global memory pointer (kernel parameter)
    This is a first-class Lean value usable in TMA operations -/
structure GPtr (dtype : GpuFloat) where
  id : VarId
  name : String  -- Original parameter name for debugging/codegen
  deriving Repr

/-- Kernel scalar value parameter
    Usable in index calculations, loop bounds, etc. -/
structure KVal (T : Type) where
  id : VarId
  name : String
  deriving Repr

/-- Get VarId from global pointer -/
def GPtr.varId {dtype : GpuFloat} (p : GPtr dtype) : VarId := p.id

/-- Get VarId from kernel value -/
def KVal.varId {T : Type} (v : KVal T) : VarId := v.id

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

/-! ## Complex Number Types -/

/-- Complex register tile (pair of real/imag tiles) -/
structure CRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  real : RT dtype rows cols layout
  imag : RT dtype rows cols layout
  deriving Repr

/-- Complex shared memory tile -/
structure CST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  real : ST dtype rows cols layout
  imag : ST dtype rows cols layout
  deriving Repr

/-- Complex register vector -/
structure CRV (dtype : GpuFloat) (len : Nat) where
  real : RV dtype len
  imag : RV dtype len
  deriving Repr

/-- Complex shared vector -/
structure CSV (dtype : GpuFloat) (len : Nat) where
  real : SV dtype len
  imag : SV dtype len
  deriving Repr

/-- Get real part VarIds -/
def CRT.realId {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : CRT dtype rows cols layout) : VarId := t.real.id

def CRT.imagId {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : CRT dtype rows cols layout) : VarId := t.imag.id

end Tyr.GPU.Codegen
