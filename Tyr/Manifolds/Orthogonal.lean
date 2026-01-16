/-
  Tyr/Manifolds/Orthogonal.lean

  Orthogonal group O(n): n×n orthogonal matrices.
  Includes SkewSymmetric matrices for tangent space representation.
-/
import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch (T)
open DifferentiableManifold

/-! ## Orthogonal Group O(n)

The orthogonal group O(n) is the special case of St(n, n):
  O(n) = { Q ∈ ℝⁿˣⁿ : QᵀQ = QQᵀ = I }

The tangent space at Q consists of matrices Q·S where S is skew-symmetric:
  T_Q O(n) = { Q·S : Sᵀ = -S }

The exponential map is: exp_Q(Q·S) = Q·exp(S)
where exp(S) is the matrix exponential.
-/

/-- Orthogonal group O(n): n×n orthogonal matrices.
    Constraint: QᵀQ = QQᵀ = I -/
structure Orthogonal (n : UInt64) where
  /-- The underlying n×n orthogonal matrix -/
  matrix : T #[n, n]

namespace Orthogonal

/-- Identity element -/
def identity (n : UInt64) : Orthogonal n :=
  ⟨torch.eye n⟩

/-- Create a random orthogonal matrix via QR decomposition. -/
def random (n : UInt64) : IO (Orthogonal n) := do
  let mat ← torch.randn #[n, n]
  let (Q, _R) := torch.linalg.qr mat
  return ⟨Q⟩

/-- Project a matrix to O(n) using QR decomposition.
    The Q factor is orthogonal. -/
def project (n : UInt64) (mat : T #[n, n]) : Orthogonal n :=
  let (Q, _R) := torch.linalg.qr mat
  ⟨Q⟩

/-- Transpose (also inverse for orthogonal matrices). -/
def transpose (Q : Orthogonal n) : Orthogonal n :=
  ⟨torch.nn.transpose2d Q.matrix⟩

/-- Matrix multiplication (group operation). -/
def mul (Q R : Orthogonal n) : Orthogonal n :=
  ⟨torch.nn.mm Q.matrix R.matrix⟩

end Orthogonal

/-- Skew-symmetric matrix: Sᵀ = -S
    Used for tangent space representation of O(n) -/
structure SkewSymmetric (n : UInt64) where
  /-- The underlying matrix (should satisfy Sᵀ = -S) -/
  matrix : T #[n, n]

namespace SkewSymmetric

/-- Zero skew-symmetric matrix -/
def zero (n : UInt64) : SkewSymmetric n :=
  ⟨torch.zeros #[n, n]⟩

/-- Create a skew-symmetric matrix from any matrix: (A - Aᵀ)/2 -/
def fromMatrix (mat : T #[n, n]) : SkewSymmetric n :=
  let transposed := torch.nn.transpose2d mat
  let skew := torch.mul_scalar (torch.sub mat transposed) 0.5
  ⟨skew⟩

/-- Add two skew-symmetric matrices -/
def add (S T : SkewSymmetric n) : SkewSymmetric n :=
  ⟨torch.add S.matrix T.matrix⟩

/-- Scale a skew-symmetric matrix -/
def smul (s : Float) (S : SkewSymmetric n) : SkewSymmetric n :=
  ⟨torch.mul_scalar S.matrix s⟩

/-- Frobenius inner product -/
def inner (S T : SkewSymmetric n) : Float :=
  torch.nn.item (torch.nn.sumAll (torch.mul S.matrix T.matrix))

end SkewSymmetric

/-- Tangent vector on the orthogonal group.
    Represented as Q·S where S is skew-symmetric.
    We store S directly since Q is known from the base point. -/
structure OrthogonalTangent (n : UInt64) where
  /-- The skew-symmetric component S such that tangent = Q·S -/
  skew : SkewSymmetric n

namespace OrthogonalTangent

/-- Zero tangent -/
def zero (n : UInt64) : OrthogonalTangent n :=
  ⟨SkewSymmetric.zero n⟩

/-- Create tangent from skew-symmetric matrix -/
def fromSkew (S : SkewSymmetric n) : OrthogonalTangent n := ⟨S⟩

/-- Add tangent vectors -/
def add (v w : OrthogonalTangent n) : OrthogonalTangent n :=
  ⟨SkewSymmetric.add v.skew w.skew⟩

/-- Scale tangent vector -/
def smul (s : Float) (v : OrthogonalTangent n) : OrthogonalTangent n :=
  ⟨SkewSymmetric.smul s v.skew⟩

/-- Get the ambient representation Q·S -/
def toAmbient (Q : Orthogonal n) (v : OrthogonalTangent n) : T #[n, n] :=
  torch.nn.mm Q.matrix v.skew.matrix

/-- Project an ambient tangent to the horizontal space -/
def fromAmbient (_Q : Orthogonal n) (Z : T #[n, n]) : OrthogonalTangent n :=
  ⟨SkewSymmetric.fromMatrix Z⟩

end OrthogonalTangent

/-- Orthogonal group as a DifferentiableManifold.
    Uses the matrix exponential for the proper exponential map: exp_Q(Q·S) = Q·exp(S). -/
instance orthogonalManifold (n : UInt64) : DifferentiableManifold (Orthogonal n) where
  Tangent _ := OrthogonalTangent n
  Cotangent _ := OrthogonalTangent n
  zeroTangent _ := OrthogonalTangent.zero n
  zeroCotangent _ := OrthogonalTangent.zero n
  addTangent v w := OrthogonalTangent.add v w
  addCotangent v w := OrthogonalTangent.add v w
  scaleTangent s v := OrthogonalTangent.smul s v
  sharp := id
  flat := id
  exp Q v :=
    -- Proper exponential map: exp_Q(Q·S) = Q · exp(S)
    -- where S is skew-symmetric, so exp(S) is orthogonal
    let expS := torch.linalg.matrix_exp v.skew.matrix
    let newQ := torch.nn.mm Q.matrix expS
    -- QR for numerical stability (exp(S) should be orthogonal, but floating point...)
    let (Q', _R) := torch.linalg.qr newQ
    ⟨Q'⟩

end Tyr.AD
