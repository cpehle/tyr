import Tyr.AD.Sparse.Add

/-!
# Tyr.AD.Sparse.Compose

Sparse composition without dense fallback.
-/

namespace Tyr.AD.Sparse

private def composeEntries
    (inEntries : Array SparseEntry)
    (outEntries : Array SparseEntry) :
    Array SparseEntry := Id.run do
  let mut out : Array SparseEntry := #[]
  for ein in inEntries do
    for eout in outEntries do
      if ein.dst = eout.src then
        out := out.push {
          src := ein.src
          dst := eout.dst
          weight := ein.weight * eout.weight
        }
  return coalesceEntries out

/-- Compose maps as `outMap ∘ inMap`. -/
def compose (inMap outMap : SparseLinearMap) : Except String SparseLinearMap := do
  validateMap inMap
  validateMap outMap
  requireComposableDims? inMap.outDim? outMap.inDim?

  let inDim? := inMap.inDim?
  let outDim? := outMap.outDim?

  if inMap.isIdentityLike then
    let mergedInDim? ← mergeDim? "input" inMap.inDim? outMap.inDim?
    pure { outMap with repr := .compose outMap.repr inMap.repr, inDim? := mergedInDim? }
  else if outMap.isIdentityLike then
    let mergedOutDim? ← mergeDim? "output" inMap.outDim? outMap.outDim?
    pure { inMap with repr := .compose outMap.repr inMap.repr, outDim? := mergedOutDim? }
  else
    pure {
      repr := .compose outMap.repr inMap.repr
      inDim? := inDim?
      outDim? := outDim?
      entries := composeEntries inMap.entries outMap.entries
    }

end Tyr.AD.Sparse
