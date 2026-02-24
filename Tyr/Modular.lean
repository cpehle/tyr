/-
  Tyr/Modular.lean

  Re-exports all modular norm components.

  The modular norm provides a principled way to normalize optimizer updates
  so that learning rates transfer across network width and depth.

  Based on "Scalable Optimization in the Modular Norm" (NeurIPS 2024).
  https://arxiv.org/abs/2405.14813

  Available components:
  - NormedModule: Typeclass for modules with norms on weight spaces
  - Atomic instances: Linear, Embedding, LayerNorm
  - Composition rules: Sequential, parallel (product), arrays
-/

import Tyr.Modular.Norm
import Tyr.Modular.Atomic
import Tyr.Modular.Compose

/-!
# `Tyr.Modular`

Composable modular-building abstraction layer for assembling reusable model components.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

