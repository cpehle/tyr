/-
  Tyr/Model/Utils.lean

  Shared model utility functions used across multiple model families.
  These are higher-level than Tyr.Torch primitives and are specific to
  neural network initialization and token-level inference helpers.

  Note: `logicalOr`, `falseMask`, `zerosOn`, `onesOn`, `castLike`,
  and `restoreInputDType` are already provided by `Tyr.Torch`.
-/
import Tyr.Torch

namespace torch.Model

open torch

/-- Kaiming-like weight init: randn scaled by sqrt(2 / fanIn). -/
def initWeight (shape : Shape) (fanIn : UInt64) : IO (T shape) := do
  let std := Float.sqrt (2.0 / fanIn.toFloat)
  let w ← torch.randn shape
  pure (autograd.set_requires_grad (mul_scalar w std) true)

/-- Zero bias tensor with requires_grad=true. -/
def initBias (shape : Shape) : T shape :=
  autograd.set_requires_grad (torch.zeros shape) true

end torch.Model
