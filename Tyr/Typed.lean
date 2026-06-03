import Tyr.Typed.DType
import Tyr.Typed.Spec
import Tyr.Typed.Device
import Tyr.Typed.Tensor

/-!
# Tyr.Typed

LeanMLX-inspired typed facade for Tyr.

The facade keeps the existing `T s` runtime tensor and adds a dtype-aware layer
for operation specs, strict boundary checks, and optional typed wrappers.
-/
