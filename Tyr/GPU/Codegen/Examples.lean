import Tyr.GPU.Kernels.Examples
import Tyr.GPU.Codegen.Notation
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Attribute

/-!
# Tyr.GPU.Codegen.Examples

Guide module for discovering and using GPU DSL examples.

This file intentionally does **not** define new kernels. Instead it points to
canonical executable examples in `Tyr.GPU.Kernels.Examples`, then summarizes
how to inspect generated code and which authoring patterns to follow.

## Canonical Example Sources

- `Tyr.GPU.Kernels.Examples`:
  canonical `@[gpu_kernel]` declarations with explicit `GPtr`/`KVal` inputs and
  explicit global-memory I/O.
- `Tyr.GPU.Kernels.FlashAttn`:
  larger end-to-end kernels for training-oriented workflows.

## Recommended Pattern

For runtime integration, prefer kernels with explicit inputs:

```lean
@[gpu_kernel .SM90]
def myKernel (xPtr : GPtr GpuFloat.Float32) (yPtr : GPtr GpuFloat.Float32)
    (outPtr : GPtr GpuFloat.Float32) (n : KVal UInt64) : KernelM Unit := do
  let _ := n
  let coord ← blockCoord2D
  ...
```

## Codegen Inspection

- Use `#print_gpu_kernel <name>` to print generated kernel source.
- Use project generation commands (`GenerateMain`, doc generation pipeline) to
  emit translation units for multiple kernels.
- `Tyr.GPU.Codegen.Notation` adds scoped sugar (`⬝`, tile `+`, etc.) on top of
  the same underlying IR-emitting ops.
- For legacy name references, use the deprecated aliases in
  `Tyr.GPU.Kernels.Examples` (`simpleGemmNew`, `flashAttnFwdNew`).
-/

namespace Tyr.GPU.Codegen.Examples

-- Quick navigation checks for canonical example declarations.
#check Tyr.GPU.Kernels.Examples.simpleGemm
#check Tyr.GPU.Kernels.Examples.simpleGemm.kernel
#check Tyr.GPU.Kernels.Examples.simpleGemm.launch
#check Tyr.GPU.Kernels.Examples.flashAttnFwd
#check Tyr.GPU.Kernels.Examples.flashAttnFwd.kernel

end Tyr.GPU.Codegen.Examples
