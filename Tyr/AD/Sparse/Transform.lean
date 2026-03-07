import Tyr.AD.Sparse.Map

/-!
# Tyr.AD.Sparse.Transform

Constructors and transform helpers for sparse maps.
-/

namespace Tyr.AD.Sparse

/-- Placeholder map used by rule scaffolding prior to shape propagation. -/
def placeholder : SparseLinearMap :=
  { repr := .placeholder }

/-- Conservative identity-like placeholder used by default local-Jac rules. -/
def identityLike : SparseLinearMap :=
  { repr := .identityLike }

/-- Sparse zero map with known shape. -/
def zeroMap (inDim outDim : DimSize) : SparseLinearMap :=
  { repr := .zero, inDim? := some inDim, outDim? := some outDim, entries := #[] }

/-- Identity map with explicit dimension and diagonal sparse entries. -/
def identityMap (n : DimSize) : SparseLinearMap :=
  {
    repr := .identity n
    inDim? := some n
    outDim? := some n
    entries := (Array.range n).map fun i => ({ src := i, dst := i, weight := 1.0 } : SparseEntry)
  }

end Tyr.AD.Sparse
