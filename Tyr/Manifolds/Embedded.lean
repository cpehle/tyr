import Tyr.Manifolds.Basic
import Tyr.Manifolds.Stiefel
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
import Tyr.Manifolds.Hyperbolic
import Tyr.Module.Derive

namespace Tyr.AD

open torch
open DifferentiableManifold

/--
Manifolds that admit a distinguished ambient representation together with
projection from ambient directions to the tangent space.
-/
class EmbeddedManifold (M : Type) [DifferentiableManifold M] where
  Ambient : Type
  toAmbient : M → Ambient
  projectAmbientTangent : {x : M} → Ambient → Tangent x

namespace EmbeddedManifold

/-- Project an ambient direction to the tangent space and retract. -/
def retractAmbientStep [DifferentiableManifold M] [EmbeddedManifold M]
    (x : M) (ambientDirection : EmbeddedManifold.Ambient (M := M)) (lr : Float) : M :=
  let tangent := EmbeddedManifold.projectAmbientTangent (x := x) ambientDirection
  let step := scaleTangent (-lr) tangent
  retract x step

end EmbeddedManifold

instance tensorEmbedded {s : Shape} : EmbeddedManifold (T s) where
  Ambient := T s
  toAmbient := id
  projectAmbientTangent := id

instance stiefelEmbedded (n p : UInt64) : EmbeddedManifold (Stiefel n p) where
  Ambient := T #[n, p]
  toAmbient := fun x => x.matrix
  projectAmbientTangent := fun {x} Z => StiefelTangent.project x Z

instance orthogonalEmbedded (n : UInt64) : EmbeddedManifold (Orthogonal n) where
  Ambient := T #[n, n]
  toAmbient := fun x => x.matrix
  projectAmbientTangent := fun {x} Z => OrthogonalTangent.fromAmbient x Z

instance grassmannEmbedded (n p : UInt64) : EmbeddedManifold (Grassmann n p) where
  Ambient := T #[n, p]
  toAmbient := fun x => x.matrix
  projectAmbientTangent := fun {x} Z => GrassmannTangent.project x Z

instance hyperbolicEmbedded (n : UInt64) : EmbeddedManifold (Hyperbolic n) where
  Ambient := T #[n + 1]
  toAmbient := fun x => x.coords
  projectAmbientTangent := fun {x} Z => HyperbolicTangent.project x Z

end Tyr.AD
