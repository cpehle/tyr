import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

/-- Positive manifold of elementwise-positive `m x n` matrices. -/
structure Positive (m n : UInt64) where
  matrix : T #[m, n]

namespace Positive

private def eps : Float := 1e-6

/-- Project to positive matrices via clamping. -/
def project (m n : UInt64) (A : T #[m, n]) : Positive m n :=
  ⟨torch.clampFloat A eps 1e12⟩

/-- Random positive point. -/
def random (m n : UInt64) : IO (Positive m n) := do
  let A ← torch.randn #[m, n]
  pure (project m n (torch.add_scalar (torch.nn.abs A) eps))

end Positive

/-- Tangent vectors on positive matrices are unconstrained ambient matrices. -/
structure PositiveTangent (m n : UInt64) where
  vec : T #[m, n]

namespace PositiveTangent

def zero (m n : UInt64) : PositiveTangent m n := ⟨torch.zeros #[m, n]⟩
def add (a b : PositiveTangent m n) : PositiveTangent m n := ⟨torch.add a.vec b.vec⟩
def smul (s : Float) (v : PositiveTangent m n) : PositiveTangent m n := ⟨torch.mul_scalar v.vec s⟩

end PositiveTangent

instance positiveManifold (m n : UInt64) : DifferentiableManifold (Positive m n) where
  Tangent _ := PositiveTangent m n
  Cotangent _ := PositiveTangent m n
  zeroTangent _ := PositiveTangent.zero m n
  zeroCotangent _ := PositiveTangent.zero m n
  addTangent a b := PositiveTangent.add a b
  addCotangent a b := PositiveTangent.add a b
  scaleTangent s v := PositiveTangent.smul s v
  sharp := by
    intro x g
    let sq := torch.mul x.matrix x.matrix
    exact ⟨torch.mul g.vec sq⟩
  flat := by
    intro x v
    let sq := torch.mul x.matrix x.matrix
    let sqSafe := torch.add_scalar sq 1e-12
    exact ⟨torch.nn.div v.vec sqSafe⟩
  exp x v :=
    let quad := torch.mul v.vec v.vec
    let denom := torch.mul_scalar x.matrix 2.0
    let second := torch.nn.div quad (torch.add_scalar denom 1e-12)
    Positive.project m n (x.matrix + v.vec + second)
  retract x v :=
    let quad := torch.mul v.vec v.vec
    let denom := torch.mul_scalar x.matrix 2.0
    let second := torch.nn.div quad (torch.add_scalar denom 1e-12)
    Positive.project m n (x.matrix + v.vec + second)

end Tyr.AD
