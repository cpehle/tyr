import Tyr.GPU.Codegen.TileIR.Type
import Tyr.GPU.Codegen.TileIR.Expr
import Tyr.GPU.Codegen.TileIR.Builder
import Tyr.GPU.Codegen.TileIR.Frontend
import Tyr.GPU.Codegen.TileIR.Passes
import Tyr.GPU.Codegen.TileIR.Render
import Tyr.GPU.Codegen.TileIR.Toolchain
import Tyr.GPU.Codegen.TileIR.Attribute
import Tyr.GPU.Codegen.TileIR.Examples

/-!
# Tyr.GPU.Codegen.TileIR

Lean-side NVIDIA TileIR backend support.

This layer is intentionally backend-first. The current milestone is:

- represent TileIR directly in Lean,
- render canonical TileIR MLIR text,
- and drive NVIDIA's TileIR compiler tools when present.

Frontend integration can be layered on later, ideally via Lean attributes and a
more ergonomic DSL that targets this backend representation.
-/
