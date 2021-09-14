import Tyr.Basic
import Tyr.Torch

namespace torch
structure Linear {n m : UInt64} :=
    (w : T #[n, m])

instance  {n m : UInt64} : differentiable (@Linear n m) := ⟨@Linear n m, fun (m : Linear) => ⟨differentiable.grad m.w⟩⟩

