/-!
# Tyr.AD.Sparse.Dim

Dimension helpers for sparse linear maps.
-/

namespace Tyr.AD.Sparse

abbrev DimSize := Nat

structure LinearShape where
  inDim : DimSize
  outDim : DimSize
  deriving Repr, Inhabited, BEq

def validDimSize (n : DimSize) : Bool :=
  0 < n

def validShape (s : LinearShape) : Bool :=
  validDimSize s.inDim && validDimSize s.outDim

def composable (inShape outShape : LinearShape) : Bool :=
  inShape.outDim = outShape.inDim

/-- Merge optional dimension metadata; fails only on conflicting known values. -/
def mergeDim? (label : String) (lhs rhs : Option DimSize) : Except String (Option DimSize) :=
  match lhs, rhs with
  | some a, some b =>
    if a = b then
      .ok (some a)
    else
      .error s!"Sparse {label}-dimension mismatch: {a} != {b}."
  | some a, none => .ok (some a)
  | none, some b => .ok (some b)
  | none, none => .ok none

/-- Composition compatibility for optional dimensions. -/
def requireComposableDims?
    (inOutDim? : Option DimSize)
    (outInDim? : Option DimSize) :
    Except String Unit :=
  match inOutDim?, outInDim? with
  | some a, some b =>
    if a = b then .ok ()
    else .error s!"Sparse compose dimension mismatch: {a} != {b}."
  | _, _ => .ok ()

end Tyr.AD.Sparse
