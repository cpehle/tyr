import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

/-- Poincare ball model with ambient coordinates in `R^n`, `||x|| < 1`. -/
structure PoincareBall (n : UInt64) where
  coords : T #[n]

namespace PoincareBall

private def eps : Float := 1e-6

/-- Project ambient vector into the open unit ball (with epsilon margin). -/
def project (n : UInt64) (v : T #[n]) : PoincareBall n :=
  let norm := torch.linalg.l2Norm v
  let maxNorm := 1.0 - eps
  if norm < maxNorm then
    ⟨v⟩
  else
    ⟨torch.mul_scalar v (maxNorm / (norm + 1e-12))⟩

/-- Random point in the unit ball. -/
def random (n : UInt64) : IO (PoincareBall n) := do
  let v ← torch.randn #[n]
  pure (project n (torch.mul_scalar v 0.5))

/-- Conformal factor lambda(x) = 2 / (1 - ||x||^2). -/
def conformalFactor (x : PoincareBall n) : Float :=
  let n2 := torch.linalg.l2Norm x.coords
  2.0 / (1.0 - n2 * n2 + 1e-12)

end PoincareBall

structure PoincareBallTangent (n : UInt64) where
  vec : T #[n]

namespace PoincareBallTangent

def zero (n : UInt64) : PoincareBallTangent n := ⟨torch.zeros #[n]⟩
def add (a b : PoincareBallTangent n) : PoincareBallTangent n := ⟨torch.add a.vec b.vec⟩
def smul (s : Float) (v : PoincareBallTangent n) : PoincareBallTangent n := ⟨torch.mul_scalar v.vec s⟩

end PoincareBallTangent

instance poincareBallManifold (n : UInt64) : DifferentiableManifold (PoincareBall n) where
  Tangent _ := PoincareBallTangent n
  Cotangent _ := PoincareBallTangent n
  zeroTangent _ := PoincareBallTangent.zero n
  zeroCotangent _ := PoincareBallTangent.zero n
  addTangent a b := PoincareBallTangent.add a b
  addCotangent a b := PoincareBallTangent.add a b
  scaleTangent s v := PoincareBallTangent.smul s v
  sharp := id
  flat := id
  exp x v := PoincareBall.project n (torch.add x.coords v.vec)
  retract x v := PoincareBall.project n (torch.add x.coords v.vec)

end Tyr.AD
