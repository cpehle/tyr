import Tyr.AutoGrad
import Tyr.AD.TensorStructSchema
import Tyr.AD.Frontend.Signature
import Tyr.AD.Frontend.Companion
import Tyr.AD.Frontend.API
import Tyr.AD.JaxprLike
import Tyr.AD.Elim

/-!
# Tyr.AD

Umbrella module for AD infrastructure in Tyr.

Current contents:
- Existing IR-level JVP/VJP machinery (`Tyr.AutoGrad`).
- `TensorStruct`-aware schema metadata plus flatten/rebuild boundary support for structured AD frontends.
- Structured frontend signature metadata for binding and reconstructing `TensorStruct` leaves across flat AD vars.
- Structured companion helpers that rebuild flat frontend gradients/pullbacks into Lean-facing values.
- A first structured frontend API layer over flat backend callbacks (`call`, `linearize`, `vjp`, scalar `grad`).
- LeanJaxpr-like IR scaffolding for elimination-based AD.
- Order-policy and AlphaGrad compatibility adapters for elimination planning.

This module is intentionally additive and does not alter existing AD behavior.
-/
