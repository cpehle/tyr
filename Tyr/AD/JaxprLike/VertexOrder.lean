import Tyr.AD.JaxprLike.Core

/-!
# Tyr.AD.JaxprLike.VertexOrder

Deterministic vertex-ID and vertex-order utilities for equation-indexed `LeanJaxpr`.
-/

namespace Tyr.AD.JaxprLike

/-- Deterministic Graphax/Tyr vertex ID (1-based) for equation index (0-based). -/
def vertexIdOfEqnIdx0 (eqnIdx0 : Nat) : Nat :=
  eqnIdx0 + 1

/-- Inverse of `vertexIdOfEqnIdx0`; returns `none` for invalid 0 vertex IDs. -/
def eqnIdx0OfVertexId? (vertexId : Nat) : Option Nat :=
  if vertexId = 0 then none else some (vertexId - 1)

/-- Deterministic forward vertex order: `[1, 2, ..., eqnCount]`. -/
def forwardVertexOrder (eqnCount : Nat) : Array Nat :=
  (Array.range eqnCount).map vertexIdOfEqnIdx0

/-- Deterministic reverse vertex order: `[eqnCount, ..., 2, 1]`. -/
def reverseVertexOrder (eqnCount : Nat) : Array Nat :=
  (forwardVertexOrder eqnCount).reverse

/-- True when `vertexId` is valid for `eqnCount` equations. -/
def isValidVertexIdForEqnCount (eqnCount : Nat) (vertexId : Nat) : Bool :=
  (0 < vertexId) && (vertexId <= eqnCount)

private def firstDuplicateVertexId? (xs : Array Nat) : Option Nat := Id.run do
  let mut seen : Std.HashSet Nat := {}
  for x in xs do
    if seen.contains x then
      return some x
    seen := seen.insert x
  return none

/--
Validate a custom vertex order against equation count:
- exact length (`eqnCount`)
- all IDs within `[1, eqnCount]`
- no duplicate IDs
-/
def validateCustomVertexOrderAgainstEqnCount
    (eqnCount : Nat)
    (order : Array Nat) :
    Except String Unit :=
  if order.size != eqnCount then
    .error s!"Invalid custom vertex order length {order.size}. Expected exactly {eqnCount}."
  else
    match order.find? (fun vertexId => !(isValidVertexIdForEqnCount eqnCount vertexId)) with
    | some bad =>
      .error s!"Invalid custom vertex ID {bad}. Expected vertex IDs in [1, {eqnCount}]."
    | none =>
      match firstDuplicateVertexId? order with
      | some dup =>
        .error s!"Invalid custom vertex order: duplicate vertex ID {dup}."
      | none =>
        .ok ()

end Tyr.AD.JaxprLike
