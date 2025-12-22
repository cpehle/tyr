/-
  Tyr/Module/Linear.lean

  Linear layer module with TensorStruct and Module instances.
-/
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Widget

namespace torch

/-- Linear layer: y = xW^T (+ b optionally)
    Weight shape is [out_features, in_features] following PyTorch convention. -/
structure Linear (in_dim out_dim : UInt64) where
  weight : T #[out_dim, in_dim]
  bias : Option (T #[out_dim]) := none
  deriving TensorStruct, ToModuleDisplay

namespace Linear

/-- Initialize a linear layer with random weights -/
def init (in_dim out_dim : UInt64) (withBias : Bool := true) : IO (Linear in_dim out_dim) := do
  -- Kaiming/He initialization: std = sqrt(2 / in_dim)
  let std := Float.sqrt (2.0 / in_dim.toFloat)
  let w ← torch.randn #[out_dim, in_dim]
  let weight := mul_scalar w std
  let weight := autograd.set_requires_grad weight true
  let bias ← if withBias then do
    let b := torch.zeros #[out_dim]
    let b := autograd.set_requires_grad b true
    pure (some b)
  else
    pure none
  pure { weight, bias }

/-- Forward pass for 2D input [batch, in_dim] -> [batch, out_dim] -/
def forward2d {in_dim out_dim batch : UInt64} (lin : Linear in_dim out_dim)
    (x : T #[batch, in_dim]) : T #[batch, out_dim] :=
  let y := linear x lin.weight
  match lin.bias with
  | some b => add y (nn.expand b #[batch, out_dim])
  | none => y

/-- Forward pass for 3D input [batch, seq, in_dim] -> [batch, seq, out_dim] -/
def forward3d {in_dim out_dim batch seq : UInt64} (lin : Linear in_dim out_dim)
    (x : T #[batch, seq, in_dim]) : T #[batch, seq, out_dim] :=
  match lin.bias with
  | some b => affine3d x lin.weight b
  | none => linear3d x lin.weight

end Linear

/-- Module instance for 2D forward pass -/
instance {in_dim out_dim batch : UInt64} :
    Module (Linear in_dim out_dim) (T #[batch, in_dim]) (T #[batch, out_dim]) where
  forward := Linear.forward2d

/-- Module instance for 3D forward pass -/
instance {in_dim out_dim batch seq : UInt64} :
    Module (Linear in_dim out_dim) (T #[batch, seq, in_dim]) (T #[batch, seq, out_dim]) where
  forward := Linear.forward3d

end torch
