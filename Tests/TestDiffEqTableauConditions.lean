/-
  Tests/TestDiffEqTableauConditions.lean

  Algebraic validation of every hand-transcribed Butcher tableau:

  - row-sum consistency: ∀ i, Σⱼ aᵢⱼ = cᵢ
  - rooted-tree order conditions on the solution weights `b` through the
    method's claimed order (exhaustively up to order 5; Dopri8 is checked
    through order 5)
  - embedded-pair semantics: every stepper computes the error estimate as
    `(b − bErr)·k`, so `bErr` must hold the *embedded solution weights*
    (sum 1, satisfying the order conditions of the embedded order) — NOT a
    diffrax-style difference vector.

  A single transcribed digit being wrong silently degrades convergence
  order; these checks pin every coefficient table algebraically.
-/
import LeanTest
import Tyr.DiffEq

open torch.DiffEq

namespace Tests.TestDiffEqTableauConditions

private def dot (x y : Array Float) : Float :=
  (x.zip y).foldl (fun acc (p : Float × Float) => acc + p.1 * p.2) 0.0

private def had (x y : Array Float) : Array Float :=
  (x.zip y).map fun p => p.1 * p.2

/-- (A·v)ᵢ = Σⱼ aᵢⱼ vⱼ (ragged rows treated as zero-padded). -/
private def matVec (a : Array (Array Float)) (v : Array Float) : Array Float :=
  a.map fun row => dot row v

private def sum (x : Array Float) : Float :=
  x.foldl (· + ·) 0.0

/-- Rooted-tree order conditions `(label, lhs, rhs)` through `order ≤ 5`. -/
private def orderConditions (a : Array (Array Float)) (b c : Array Float)
    (order : Nat) : Array (String × Float × Float) := Id.run do
  let ac := matVec a c
  let c2 := had c c
  let c3 := had c2 c
  let c4 := had c3 c
  let mut conds : Array (String × Float × Float) := #[]
  if order ≥ 1 then
    conds := conds.push ("Σb = 1", sum b, 1.0)
  if order ≥ 2 then
    conds := conds.push ("b·c = 1/2", dot b c, 1.0 / 2.0)
  if order ≥ 3 then
    conds := conds ++ #[
      ("b·c² = 1/3", dot b c2, 1.0 / 3.0),
      ("b·(Ac) = 1/6", dot b ac, 1.0 / 6.0)]
  if order ≥ 4 then
    conds := conds ++ #[
      ("b·c³ = 1/4", dot b c3, 1.0 / 4.0),
      ("b·(c⊙Ac) = 1/8", dot b (had c ac), 1.0 / 8.0),
      ("b·(Ac²) = 1/12", dot b (matVec a c2), 1.0 / 12.0),
      ("b·(AAc) = 1/24", dot b (matVec a ac), 1.0 / 24.0)]
  if order ≥ 5 then
    conds := conds ++ #[
      ("b·c⁴ = 1/5", dot b c4, 1.0 / 5.0),
      ("b·(c²⊙Ac) = 1/10", dot b (had c2 ac), 1.0 / 10.0),
      ("b·(c⊙Ac²) = 1/15", dot b (had c (matVec a c2)), 1.0 / 15.0),
      ("b·(c⊙AAc) = 1/30", dot b (had c (matVec a ac)), 1.0 / 30.0),
      ("b·(Ac⊙Ac) = 1/20", dot b (had ac ac), 1.0 / 20.0),
      ("b·(Ac³) = 1/20", dot b (matVec a c3), 1.0 / 20.0),
      ("b·(A(c⊙Ac)) = 1/40", dot b (matVec a (had c ac)), 1.0 / 40.0),
      ("b·(AAc²) = 1/60", dot b (matVec a (matVec a c2)), 1.0 / 60.0),
      ("b·(AAAc) = 1/120", dot b (matVec a (matVec a ac)), 1.0 / 120.0)]
  return conds

private def tol : Float := 1e-9

/-- Validate one tableau: row sums, solution order conditions through
    `min claimed 5`, and (if present) embedded weights through `embOrder`. -/
private def checkTableau {s : Nat} (name : String) (t : ButcherTableau s)
    (embOrder : Nat := 0) : IO Unit := do
  let a := t.a.toArray
  let b := t.b.toArray
  let c := t.c.toArray
  -- row-sum consistency
  for i in [:a.size] do
    let rs := sum (a.getD i #[])
    let ci := c.getD i 0.0
    LeanTest.assertTrue ((rs - ci).abs < tol)
      s!"{name}: row {i} sum {rs} ≠ c[{i}] = {ci}"
  -- solution-weight order conditions
  let p := Nat.min t.order 5
  for (label, lhs, rhs) in orderConditions a b c p do
    LeanTest.assertTrue ((lhs - rhs).abs < tol)
      s!"{name}: order condition {label} violated: got {lhs}"
  -- embedded weights: stepper semantics are yError = (b − bErr)·k, so bErr
  -- must itself be a consistent solution-weight vector of the embedded order
  match t.bErr with
  | none => pure ()
  | some bErr =>
    let bHat := bErr.toArray
    LeanTest.assertTrue ((sum b - sum bHat).abs < tol)
      s!"{name}: Σ(b − bErr) = {sum b - sum bHat} ≠ 0 — bErr is not an embedded weight vector (difference convention?)"
    let pe := Nat.min embOrder 5
    for (label, lhs, rhs) in orderConditions a bHat c pe do
      LeanTest.assertTrue ((lhs - rhs).abs < tol)
        s!"{name}: embedded condition {label} violated: got {lhs}"

@[test]
def testExplicitTableauConditions : IO Unit := do
  checkTableau "ralston" ralstonTableau
  checkTableau "bosh3" bosh3Tableau (embOrder := 2)
  checkTableau "rk4" rk4Tableau
  checkTableau "dopri5" dopri5Tableau (embOrder := 4)
  checkTableau "tsit5" tsit5Tableau (embOrder := 4)
  checkTableau "dopri8" dopri8Tableau (embOrder := 5)

@[test]
def testImplicitTableauConditions : IO Unit := do
  checkTableau "kvaerno3" kvaerno3Tableau (embOrder := 2)
  checkTableau "kvaerno4" kvaerno4Tableau (embOrder := 3)
  checkTableau "kvaerno5" kvaerno5Tableau (embOrder := 4)

@[test]
def testImexTableauConditions : IO Unit := do
  checkTableau "kencarp3.explicit" kencarp3Explicit (embOrder := 2)
  checkTableau "kencarp3.implicit" kencarp3Implicit (embOrder := 2)
  checkTableau "kencarp4.explicit" kencarp4Explicit (embOrder := 3)
  checkTableau "kencarp4.implicit" kencarp4Implicit (embOrder := 3)
  checkTableau "kencarp5.explicit" kencarp5Explicit (embOrder := 4)
  checkTableau "kencarp5.implicit" kencarp5Implicit (embOrder := 4)

end Tests.TestDiffEqTableauConditions
