/-
  Tyr/Module/LayerNorm.lean

  Layer Normalization module with TensorStruct and Module instances.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Widget

namespace torch

/-- Layer Normalization: normalizes over the last dimension.
    y = (x - mean) / sqrt(var + eps) * weight + bias -/
structure LayerNorm (dim : UInt64) where
  weight : T #[dim]
  bias : T #[dim]
  eps : Static Float := ⟨1e-5⟩
  deriving TensorStruct, ToModuleDisplay

namespace LayerNorm

/-- Initialize layer norm with ones for weight and zeros for bias -/
def init (dim : UInt64) (eps : Float := 1e-5) : LayerNorm dim :=
  let weight := autograd.set_requires_grad (torch.ones #[dim]) true
  let bias := autograd.set_requires_grad (torch.zeros #[dim]) true
  { weight, bias, eps := ⟨eps⟩ }

/-- Forward pass for 3D input [batch, seq, dim] -/
def forward3d {dim batch seq : UInt64} (ln : LayerNorm dim)
    (x : T #[batch, seq, dim]) : T #[batch, seq, dim] :=
  nn.layer_norm x ln.weight ln.bias ln.eps.val

end LayerNorm

/-- Module instance for 3D forward pass -/
instance {dim batch seq : UInt64} :
    Module (LayerNorm dim) (T #[batch, seq, dim]) (T #[batch, seq, dim]) where
  forward := LayerNorm.forward3d

end torch
