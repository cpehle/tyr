import Tyr.AD.Elim.OrderPolicy

/-!
# Tyr.AD.Elim.ConstraintFeasibility

Constraint feasibility checks for elimination scheduling.
Current version focuses on hard precedence edges.
-/

namespace Tyr.AD.Elim

/-- Predicate for whether a vertex has already been eliminated. -/
abbrev EliminatedPred := VertexId1 → Bool

/--
Hard precedence `(u, v)` is satisfied for candidate `v` iff `u` is already eliminated.
For any candidate other than `v`, this edge does not constrain feasibility.
-/
def hardEdgeFeasible (isEliminated : EliminatedPred) (candidate : VertexId1) (edge : VertexId1 × VertexId1) : Bool :=
  let (u, v) := edge
  if candidate == v then
    isEliminated u
  else
    true

/-- Check hard precedence feasibility for a candidate vertex. -/
def hardConstraintsFeasible
    (constraints : ConstraintSpec)
    (isEliminated : EliminatedPred)
    (candidate : VertexId1) :
    Bool :=
  constraints.hardPrecedence.all (hardEdgeFeasible isEliminated candidate)

/--
Overall step feasibility:
- candidate must not already be eliminated
- hard constraints must be satisfied
-/
def constraintFeasible
    (constraints : ConstraintSpec)
    (isEliminated : EliminatedPred)
    (candidate : VertexId1) :
    Bool :=
  !(isEliminated candidate) && hardConstraintsFeasible constraints isEliminated candidate

/-- Compute feasible candidates from a 1-based vertex list. -/
def filterFeasibleCandidates
    (constraints : ConstraintSpec)
    (isEliminated : EliminatedPred)
    (candidates : Array VertexId1) :
    Array VertexId1 :=
  candidates.filter (constraintFeasible constraints isEliminated ·)

end Tyr.AD.Elim
