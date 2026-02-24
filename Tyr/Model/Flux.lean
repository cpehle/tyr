/-
  Tyr/Model/Flux.lean

  Flux diffusion model for image generation.
  Re-exports all Flux components.
-/

import Tyr.Model.Flux.Config
import Tyr.Model.Flux.RoPE
import Tyr.Model.Flux.Modulation
import Tyr.Model.Flux.QKNorm
import Tyr.Model.Flux.SingleStreamBlock
import Tyr.Model.Flux.DoubleStreamBlock
import Tyr.Model.Flux.Model
import Tyr.Model.Flux.Sampling
import Tyr.Model.Flux.Weights

/-!
# `Tyr.Model.Flux`

Flux model entrypoint that re-exports Flux architecture components and weight-loading helpers.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

