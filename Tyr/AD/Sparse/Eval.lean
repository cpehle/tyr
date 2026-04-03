import Tyr.AD.Sparse.Validate

/-!
# Tyr.AD.Sparse.Eval

Executable numeric interpretation for sparse linear maps.
-/

namespace Tyr.AD.Sparse

private def requireKnownDim
    (label : String)
    (dim? : Option DimSize) :
    Except String DimSize :=
  match dim? with
  | some dim =>
      if validDimSize dim then
        .ok dim
      else
        .error s!"Sparse {label} dimension must be positive, got {dim}."
  | none =>
      .error s!"Sparse {label} dimension is required for numeric evaluation."

private def requireExecutableRepr
    (m : SparseLinearMap) :
    Except String Unit :=
  match m.repr with
  | .placeholder =>
      .error "Cannot numerically evaluate placeholder sparse maps."
  | .identityLike =>
      .error "Cannot numerically evaluate identity-like placeholder sparse maps."
  | _ =>
      .ok ()

private def requireVectorWidth
    (label : String)
    (xs : Array Float)
    (expected : Nat) :
    Except String Unit :=
  if xs.size = expected then
    .ok ()
  else
    .error s!"Sparse {label} width mismatch: expected {expected}, got {xs.size}."

/--
Apply a sparse linear map in forward/JVP mode: `y = J x`.

The input width must match `inDim?`, and the returned output width is `outDim?`.
Placeholder maps are rejected because they do not carry executable numeric
semantics.
-/
def SparseLinearMap.apply
    (m : SparseLinearMap)
    (input : Array Float) :
    Except String (Array Float) := do
  validateMap m
  requireExecutableRepr m
  let inDim ← requireKnownDim "input" m.inDim?
  let outDim ← requireKnownDim "output" m.outDim?
  requireVectorWidth "input vector" input inDim

  let mut out := Array.replicate outDim 0.0
  for entry in m.entries do
    let accum := out.getD entry.dst 0.0
    out := out.set! entry.dst (accum + entry.weight * input[entry.src]!)
  pure out

/--
Apply a sparse linear map in reverse/VJP mode: `x̄ = Jᵀ ȳ`.

The cotangent width must match `outDim?`, and the returned input cotangent width
is `inDim?`.
-/
def SparseLinearMap.pullback
    (m : SparseLinearMap)
    (outputCotangent : Array Float) :
    Except String (Array Float) := do
  validateMap m
  requireExecutableRepr m
  let inDim ← requireKnownDim "input" m.inDim?
  let outDim ← requireKnownDim "output" m.outDim?
  requireVectorWidth "output cotangent" outputCotangent outDim

  let mut out := Array.replicate inDim 0.0
  for entry in m.entries do
    let accum := out.getD entry.src 0.0
    out := out.set! entry.src (accum + entry.weight * outputCotangent[entry.dst]!)
  pure out

/-- Materialize a sparse linear map into row-major dense form for tests/debugging. -/
def SparseLinearMap.toDenseRows
    (m : SparseLinearMap) :
    Except String (Array (Array Float)) := do
  validateMap m
  requireExecutableRepr m
  let inDim ← requireKnownDim "input" m.inDim?
  let outDim ← requireKnownDim "output" m.outDim?

  let mut rows := Array.replicate outDim (Array.replicate inDim 0.0)
  for entry in m.entries do
    let row := rows[entry.dst]!
    let accum := row.getD entry.src 0.0
    rows := rows.set! entry.dst (row.set! entry.src (accum + entry.weight))
  pure rows

end Tyr.AD.Sparse
