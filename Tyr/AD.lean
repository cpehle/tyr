import Tyr.AutoGrad
import Tyr.AD.JaxprLike
import Tyr.AD.Elim

/-!
# Tyr.AD

Umbrella module for AD infrastructure in Tyr.

Current contents:
- Existing IR-level JVP/VJP machinery (`Tyr.AutoGrad`).
- LeanJaxpr-like IR scaffolding for elimination-based AD.
- Order-policy and AlphaGrad compatibility adapters for elimination planning.

This module is intentionally additive and does not alter existing AD behavior.
-/
