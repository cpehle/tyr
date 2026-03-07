import Tyr.Modular.Norm
import Tyr.Modular.Atomic
import Tyr.Modular.Compose
import Tyr.Modular.Budget
import Tyr.Modular.MetricFactor
import Tyr.Modular.Manifold
import Tyr.Modular.RiemannianModule

/-!
# Tyr.Modular

`Tyr.Modular` is the umbrella import for modular-norm abstractions in Tyr.
It re-exports the components used to define normed modules and compose them in a way
that supports width/depth-robust optimization scaling.

## Major Components

- `Norm`: foundational modular-norm abstractions.
- `Atomic`: atomic module instances (for example linear/layer-style primitives).
- `Compose`: composition rules for sequential/product/container module structures.
- `Budget`: modular sensitivity -> LR budget compilation utilities.
- `Manifold`: manifold-native module wrappers and modular-to-optimizer bridges.

## Scope

This module collects the modular-norm stack behind one import.
It targets optimizer/module scaling workflows rather than general tensor math.
-/
