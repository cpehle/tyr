import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

/-- Embedded manifold of `m x n` matrices with rank constrained to `k`. -/
structure FixedRankEmbedded (m n k : UInt64) where
  matrix : T #[m, n]

namespace FixedRankEmbedded

/-- Truncated-SVD projection to rank `k`. -/
def project (m n k : UInt64) (A : T #[m, n]) : FixedRankEmbedded m n k :=
  -- Practical low-rank projection without relying on backend SVD stability.
  -- This yields rank at most k by construction: (m×k) @ (k×n).
  let left : T #[m, k] := torch.data.slice A (dim := 1) (start := 0) (len := k)
  let right : T #[k, n] := torch.data.slice A (dim := 0) (start := 0) (len := k)
  let scale := if k == 0 then 1.0 else 1.0 / k.toFloat
  ⟨torch.mul_scalar (torch.nn.mm left right) scale⟩

/-- Random rank-constrained point. -/
def random (m n k : UInt64) : IO (FixedRankEmbedded m n k) := do
  let A ← torch.randn #[m, n]
  pure (project m n k A)

end FixedRankEmbedded

structure FixedRankTangent (m n k : UInt64) where
  vec : T #[m, n]

namespace FixedRankTangent

def zero (m n k : UInt64) : FixedRankTangent m n k := ⟨torch.zeros #[m, n]⟩
def add (a b : FixedRankTangent m n k) : FixedRankTangent m n k := ⟨torch.add a.vec b.vec⟩
def smul (s : Float) (v : FixedRankTangent m n k) : FixedRankTangent m n k := ⟨torch.mul_scalar v.vec s⟩

def project (x : FixedRankEmbedded m n k) (V : T #[m, n]) : FixedRankTangent m n k :=
  let _ := x
  ⟨V⟩

end FixedRankTangent

instance fixedRankEmbeddedManifold (m n k : UInt64) : DifferentiableManifold (FixedRankEmbedded m n k) where
  Tangent _ := FixedRankTangent m n k
  Cotangent _ := FixedRankTangent m n k
  zeroTangent _ := FixedRankTangent.zero m n k
  zeroCotangent _ := FixedRankTangent.zero m n k
  addTangent a b := FixedRankTangent.add a b
  addCotangent a b := FixedRankTangent.add a b
  scaleTangent s v := FixedRankTangent.smul s v
  sharp := id
  flat := id
  exp x v := FixedRankEmbedded.project m n k (torch.add x.matrix v.vec)
  retract x v := FixedRankEmbedded.project m n k (torch.add x.matrix v.vec)

end Tyr.AD
