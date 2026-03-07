import Tyr.AD.Sparse.Validate

/-!
# Tyr.AD.Sparse.Add

Sparse additive merge and coalescing.
-/

namespace Tyr.AD.Sparse

private def entryLt (a b : SparseEntry) : Bool :=
  if a.src = b.src then a.dst < b.dst else a.src < b.src

private def sortEntries (entries : Array SparseEntry) : Array SparseEntry :=
  (entries.toList.mergeSort entryLt).toArray

/-- Coalesce duplicate `(src,dst)` entries by summing weights and dropping zeros. -/
def coalesceEntries (entries : Array SparseEntry) : Array SparseEntry := Id.run do
  let sorted := sortEntries entries
  let mut out : Array SparseEntry := #[]
  for e in sorted do
    match out.back? with
    | some last =>
      if last.src = e.src && last.dst = e.dst then
        let merged : SparseEntry := { src := last.src, dst := last.dst, weight := last.weight + e.weight }
        out := out.pop.push merged
      else
        out := out.push e
    | none =>
      out := out.push e
  return out.filter (fun e => Float.abs e.weight > 1e-12)

/-- Add two sparse maps with strict shape compatibility checks. -/
def add (lhs rhs : SparseLinearMap) : Except String SparseLinearMap := do
  validateMap lhs
  validateMap rhs
  let inDim? ← mergeDim? "input" lhs.inDim? rhs.inDim?
  let outDim? ← mergeDim? "output" lhs.outDim? rhs.outDim?
  pure {
    repr := .add lhs.repr rhs.repr
    inDim? := inDim?
    outDim? := outDim?
    entries := coalesceEntries (lhs.entries ++ rhs.entries)
  }

end Tyr.AD.Sparse
