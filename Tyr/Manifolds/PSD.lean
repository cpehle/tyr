import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

/-- Quotient-factor representation for fixed-rank PSD matrices via `Y in R^{n x k}`. -/
structure PSDFixedRank (n k : UInt64) where
  factor : T #[n, k]

namespace PSDFixedRank

/-- Ambient matrix in the PSD cone represented by this factor. -/
def toPSD (x : PSDFixedRank n k) : T #[n, n] :=
  torch.nn.mm x.factor (torch.nn.transpose2d x.factor)

/-- Random point (factor matrix). -/
def random (n k : UInt64) : IO (PSDFixedRank n k) := do
  let y ← torch.randn #[n, k]
  pure ⟨y⟩

end PSDFixedRank

structure PSDFixedRankTangent (n k : UInt64) where
  vec : T #[n, k]

namespace PSDFixedRankTangent

def zero (n k : UInt64) : PSDFixedRankTangent n k := ⟨torch.zeros #[n, k]⟩
def add (a b : PSDFixedRankTangent n k) : PSDFixedRankTangent n k := ⟨torch.add a.vec b.vec⟩
def smul (s : Float) (v : PSDFixedRankTangent n k) : PSDFixedRankTangent n k := ⟨torch.mul_scalar v.vec s⟩

/-- Horizontal projection proxy akin Stiefel projection in factor space. -/
def project (x : PSDFixedRank n k) (V : T #[n, k]) : PSDFixedRankTangent n k :=
  let YtV := torch.nn.mm (torch.nn.transpose2d x.factor) V
  let sym := torch.mul_scalar (torch.add YtV (torch.nn.transpose2d YtV)) 0.5
  ⟨torch.sub V (torch.nn.mm x.factor sym)⟩

end PSDFixedRankTangent

instance psdFixedRankManifold (n k : UInt64) : DifferentiableManifold (PSDFixedRank n k) where
  Tangent _ := PSDFixedRankTangent n k
  Cotangent _ := PSDFixedRankTangent n k
  zeroTangent _ := PSDFixedRankTangent.zero n k
  zeroCotangent _ := PSDFixedRankTangent.zero n k
  addTangent a b := PSDFixedRankTangent.add a b
  addCotangent a b := PSDFixedRankTangent.add a b
  scaleTangent s v := PSDFixedRankTangent.smul s v
  sharp := id
  flat := id
  exp x v := ⟨torch.add x.factor v.vec⟩
  retract x v := ⟨torch.add x.factor v.vec⟩

/-- Elliptope factor manifold: row-wise unit norms in `R^{n x k}`. -/
structure Elliptope (n k : UInt64) where
  factor : T #[n, k]

namespace Elliptope

private def normalizeRows {n k : UInt64} (Y : T #[n, k]) : T #[n, k] :=
  let norms := torch.linalg.rowNorms Y
  let safeNorms := torch.add_scalar norms 1e-12
  let normsCol : T #[n, 1] := torch.reshape safeNorms #[n, 1]
  let expanded : T #[n, k] := torch.nn.expand normsCol #[n, k]
  torch.nn.div Y expanded

/-- Project ambient matrix to row-normalized elliptope factor. -/
def project (n k : UInt64) (Y : T #[n, k]) : Elliptope n k :=
  ⟨normalizeRows Y⟩

/-- Random row-normalized factor. -/
def random (n k : UInt64) : IO (Elliptope n k) := do
  let y ← torch.randn #[n, k]
  pure (project n k y)

end Elliptope

structure ElliptopeTangent (n k : UInt64) where
  vec : T #[n, k]

namespace ElliptopeTangent

def zero (n k : UInt64) : ElliptopeTangent n k := ⟨torch.zeros #[n, k]⟩
def add (a b : ElliptopeTangent n k) : ElliptopeTangent n k := ⟨torch.add a.vec b.vec⟩
def smul (s : Float) (v : ElliptopeTangent n k) : ElliptopeTangent n k := ⟨torch.mul_scalar v.vec s⟩

/-- Project ambient direction to row-wise tangent space (`<y_i, v_i> = 0`). -/
def project (x : Elliptope n k) (V : T #[n, k]) : ElliptopeTangent n k :=
  let onesCol : T #[k, 1] := torch.ones #[k, 1]
  let rowDotsCol : T #[n, 1] := torch.nn.mm (torch.mul x.factor V) onesCol
  let rowDots : T #[n, k] := torch.nn.expand rowDotsCol #[n, k]
  ⟨torch.sub V (torch.mul x.factor rowDots)⟩

end ElliptopeTangent

instance elliptopeManifold (n k : UInt64) : DifferentiableManifold (Elliptope n k) where
  Tangent _ := ElliptopeTangent n k
  Cotangent _ := ElliptopeTangent n k
  zeroTangent _ := ElliptopeTangent.zero n k
  zeroCotangent _ := ElliptopeTangent.zero n k
  addTangent a b := ElliptopeTangent.add a b
  addCotangent a b := ElliptopeTangent.add a b
  scaleTangent s v := ElliptopeTangent.smul s v
  sharp := id
  flat := id
  exp x v := Elliptope.project n k (torch.add x.factor v.vec)
  retract x v := Elliptope.project n k (torch.add x.factor v.vec)

end Tyr.AD
