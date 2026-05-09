import Tyr.Basic
import Tyr.Torch 

namespace torch

structure Affine {n m : UInt64} where
    w : T #[m, n]
    b : T #[n]

instance {n m : UInt64}: differentiable (@Affine n m) := ⟨@Affine n m, fun (a : Affine) => ⟨differentiable.grad a.w, differentiable.grad a.b⟩⟩

def Affine.step {b n m : UInt64} (a : @torch.Affine n m) (input : torch.T #[b, m]) :  torch.T #[b, n] :=
    @torch.affine b n m a.w a.b input

/-- Typed `step` — preserves the input's `TensorMeta` through the affine
    transform. Body delegates to the legacy `step`. -/
@[inline] def Affine.stepT {tm : TensorMeta} {b n m : UInt64}
    (a : @torch.Affine n m) (input : torch.Tensor tm #[b, m])
    : torch.Tensor tm #[b, n] :=
  torch.Tensor.unsafeOfT tm (a.step (torch.Tensor.toT input))
