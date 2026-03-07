import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

/-- Oblique manifold OB(m,n): matrices with unit-norm columns. -/
structure Oblique (m n : UInt64) where
  matrix : T #[m, n]

namespace Oblique

private def normalizeColumns {m n : UInt64} (A : T #[m, n]) : T #[m, n] :=
  let AtA := torch.nn.mm (torch.nn.transpose2d A) A
  let colNorms := torch.nn.sqrt (torch.linalg.diag AtA)
  let safeNorms := torch.add_scalar colNorms 1e-12
  let rowNorms : T #[1, n] := torch.reshape safeNorms #[1, n]
  let expanded : T #[m, n] := torch.nn.expand rowNorms #[m, n]
  torch.nn.div A expanded

/-- Project ambient matrix to the oblique manifold by column normalization. -/
def project (m n : UInt64) (A : T #[m, n]) : Oblique m n :=
  ⟨normalizeColumns A⟩

/-- Random point on the oblique manifold. -/
def random (m n : UInt64) : IO (Oblique m n) := do
  let A ← torch.randn #[m, n]
  pure (project m n A)

end Oblique

/-- Tangent vector on OB(m,n). -/
structure ObliqueTangent (m n : UInt64) where
  vec : T #[m, n]

namespace ObliqueTangent

/-- Project ambient direction to tangent space at `X`. -/
def project (X : Oblique m n) (V : T #[m, n]) : ObliqueTangent m n :=
  let onesCol : T #[m, 1] := torch.ones #[m, 1]
  let colDotsCol : T #[n, 1] :=
    torch.nn.mm (torch.nn.transpose2d (torch.mul X.matrix V)) onesCol
  let colDotsRow : T #[1, n] := torch.reshape colDotsCol #[1, n]
  let colDots : T #[m, n] := torch.nn.expand colDotsRow #[m, n]
  ⟨torch.sub V (torch.mul X.matrix colDots)⟩

/-- Zero tangent vector. -/
def zero (m n : UInt64) : ObliqueTangent m n := ⟨torch.zeros #[m, n]⟩

/-- Add tangent vectors. -/
def add (a b : ObliqueTangent m n) : ObliqueTangent m n :=
  ⟨torch.add a.vec b.vec⟩

/-- Scale tangent vector. -/
def smul (s : Float) (v : ObliqueTangent m n) : ObliqueTangent m n :=
  ⟨torch.mul_scalar v.vec s⟩

end ObliqueTangent

instance obliqueManifold (m n : UInt64) : DifferentiableManifold (Oblique m n) where
  Tangent _ := ObliqueTangent m n
  Cotangent _ := ObliqueTangent m n
  zeroTangent _ := ObliqueTangent.zero m n
  zeroCotangent _ := ObliqueTangent.zero m n
  addTangent a b := ObliqueTangent.add a b
  addCotangent a b := ObliqueTangent.add a b
  scaleTangent s v := ObliqueTangent.smul s v
  sharp := id
  flat := id
  exp x v :=
    Oblique.project m n (torch.add x.matrix v.vec)
  retract x v :=
    Oblique.project m n (torch.add x.matrix v.vec)

end Tyr.AD
