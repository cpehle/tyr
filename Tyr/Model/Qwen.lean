/-
  Tyr/Model/Qwen.lean

  Qwen3 model implementation for Flux text encoding.
  Re-exports all Qwen components.
-/

import Tyr.Model.Qwen.Config
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen.Attention
import Tyr.Model.Qwen.MLP
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.Embedder
import Tyr.Model.Qwen.Weights

/-!
# `Tyr.Model.Qwen`

Qwen model entrypoint that re-exports attention blocks, layers, configuration, and weight helpers.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

