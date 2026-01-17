/-
  Tyr/GPU/Codegen/Arch/Polymorphic.lean

  Polymorphic kernel generation that compiles one kernel definition
  to multiple architecture targets with automatic optimization.
-/
import Tyr.GPU.Codegen.Arch.Level
import Tyr.GPU.Codegen.Arch.Monad
import Tyr.GPU.Codegen.Arch.Capabilities
import Tyr.GPU.Codegen.Arch.Ops
import Tyr.GPU.Codegen.EmitNew

namespace Tyr.GPU.Codegen.Arch

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## Polymorphic Kernel Definition

A PolyKernel is a kernel that can generate optimized code for multiple
GPU architectures from a single definition.
-/

/-- A kernel body that can be instantiated for any architecture -/
structure PolyKernelBody where
  /-- Generate the kernel body for a specific architecture -/
  generate : (arch : ArchLevel) → KernelM Unit

/-- A kernel parameterized by target architecture -/
structure PolyKernel where
  /-- Kernel name -/
  name : String
  /-- Kernel parameters -/
  params : Array KParam
  /-- Architecture-polymorphic kernel body -/
  body : PolyKernelBody

/-- Result of compiling a PolyKernel to multiple architectures -/
structure CompiledPolyKernel where
  /-- Original kernel name -/
  name : String
  /-- Kernels compiled for each target architecture -/
  kernels : Array (ArchLevel × Kernel)
  /-- Combined C++ source with preprocessor guards -/
  cppSource : String
  deriving Repr

namespace PolyKernel

/-- Compile the kernel for a single architecture -/
def compileForArch (k : PolyKernel) (arch : ArchLevel) : Kernel :=
  buildKernelM k.name arch.toGpuArch k.params (k.body.generate arch)

/-- Compile to all architectures -/
def compileAll (k : PolyKernel) : Array (ArchLevel × Kernel) :=
  ArchLevel.all.map fun arch => (arch, k.compileForArch arch)

/-- Compile to specified architectures -/
def compileFor (k : PolyKernel) (archs : Array ArchLevel) : Array (ArchLevel × Kernel) :=
  archs.map fun arch => (arch, k.compileForArch arch)

/-- Generate C++ code with architecture preprocessor guards -/
def emitCpp (k : PolyKernel) (archs : Array ArchLevel := ArchLevel.all) : String :=
  let header := "#include <kittens.cuh>\nusing namespace kittens;\n\n"
  let body := archs.foldl (fun acc arch =>
    let kernel := k.compileForArch arch
    let code := generateKernel kernel
    -- Extract just the kernel function body (skip headers for subsequent arches)
    let bodyLines := code.splitOn "\n" |>.filter fun line =>
      !line.startsWith "#include" && !line.startsWith "using namespace"
    let bodyCode := String.intercalate "\n" bodyLines
    acc ++ s!"#if defined({arch.toGuard})\n{bodyCode}#endif // {arch.toGuard}\n\n"
  ) ""
  header ++ body

/-- Full compilation: returns compiled kernels and C++ source -/
def compile (k : PolyKernel) (archs : Array ArchLevel := ArchLevel.all) : CompiledPolyKernel := {
  name := k.name
  kernels := k.compileFor archs
  cppSource := k.emitCpp archs
}

end PolyKernel

/-! ## Registry for Polymorphic Kernels -/

/-- Registered polymorphic kernel (stored in environment) -/
structure RegisteredPolyKernel where
  /-- Original kernel name -/
  name : Name
  /-- Target architectures -/
  archs : Array ArchLevel
  /-- Compiled kernels (one per architecture) -/
  kernels : Array Kernel
  /-- Generated C++ code for each architecture -/
  cppCodes : Array String
  /-- Combined C++ with guards -/
  combinedCpp : String

/-! ## Helper Functions for Building Poly Kernels -/

/-- Create a PolyKernel from a simple generator function -/
def mkPolyKernel (name : String) (params : Array KParam)
    (gen : ArchLevel → KernelM Unit) : PolyKernel := {
  name := name
  params := params
  body := { generate := gen }
}

/-- Create a PolyKernel using architecture config -/
def mkPolyKernelWithConfig (name : String) (params : Array KParam)
    (gen : (arch : ArchLevel) → ArchCapabilitiesRecord → KernelM Unit) : PolyKernel := {
  name := name
  params := params
  body := { generate := fun arch => gen arch arch.capabilities }
}

/-- Create a PolyKernel from an ArchKernelM generator -/
def mkArchPolyKernel (name : String) (params : Array KParam)
    (gen : (arch : ArchLevel) → ArchKernelM arch Unit) : PolyKernel := {
  name := name
  params := params
  body := { generate := fun arch => (gen arch).run }
}

/-! ## Example Usage Patterns

The recommended way to define polymorphic kernels is using the @[gpu_kernel] attribute
without an architecture argument. The attribute detects that the function takes an
ArchLevel parameter and generates kernels for all architectures automatically.

Example:
```lean
@[gpu_kernel]  -- No arch = polymorphic, generates for SM80/SM90/SM100
def myPolyKernel (input : GPtr .BFloat16) (output : GPtr .BFloat16)
    (arch : ArchLevel) : ArchKernelM arch Unit := do
  -- Use arch-specific operations via typeclasses
  ...

-- Auto-generated:
-- myPolyKernel.kernel_SM80 : Kernel
-- myPolyKernel.kernel_SM90 : Kernel
-- myPolyKernel.kernel_SM100 : Kernel
-- myPolyKernel.launch_SM80 : ... → IO Unit
-- myPolyKernel.launch_SM90 : ... → IO Unit
-- myPolyKernel.launch_SM100 : ... → IO Unit
-- myPolyKernel.launch : ArchLevel → ... → IO Unit  (runtime dispatch)
```

For manual PolyKernel construction (without attribute), use mkPolyKernel:
-/

/-- Example: Manual PolyKernel construction for cases where the attribute isn't suitable.
    This shows how to build a PolyKernel that dispatches to architecture-specific code. -/
def exampleManualPolyKernel : PolyKernel := mkPolyKernel
  "example_manual_poly"
  #[
    { name := "input", dtype := .BFloat16, isPointer := true },
    { name := "output", dtype := .BFloat16, isPointer := true }
  ]
  fun arch => do
    -- Create parameter bindings (VarIds 0, 1 match params array)
    let _input : GPtr GpuFloat.BFloat16 := ⟨⟨0⟩, "input"⟩
    let _output : GPtr GpuFloat.BFloat16 := ⟨⟨1⟩, "output"⟩

    let cfg := arch.capabilities
    let (tileM, tileN, _) := cfg.mmaTileSize

    comment s!"=== Architecture: {arch} ==="
    comment s!"Tile size: {tileM}x{tileN}, TMA: {cfg.hasTMA}, WGMMA: {cfg.hasWGMMA}"

    -- Allocate tiles
    let a ← Codegen.allocRT .BFloat16 tileM tileN
    let b ← Codegen.allocRT .BFloat16 tileM tileN
    let c ← Codegen.zeroRT .Float32 tileM tileN

    Codegen.add a a b
    Codegen.convert c a

/-! ## Multi-Architecture Code Generation Utilities -/

/-- Generate unified runtime launcher that dispatches based on architecture -/
def generateUnifiedLauncher (name : String) (params : Array KParam)
    (archs : Array ArchLevel) : String :=
  let paramDecls := params.toList.map fun p =>
    if p.isPointer then s!"const {p.dtype.toCpp}* {p.name}"
    else s!"{p.dtype.toCpp} {p.name}"
  let paramStr := String.intercalate ", " paramDecls
  let argStr := String.intercalate ", " (params.toList.map (·.name))

  let archCases := archs.toList.map fun arch =>
    let suffix := arch.toSuffix
    s!"    case {arch.toNat}: {name}{suffix}<<<grid, block, smem, stream>>>({argStr}); break;\n"

  "\nvoid " ++ name ++ "_launch(int arch, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, " ++
  paramStr ++ ") {\n  switch(arch) {\n" ++
  String.join archCases ++
  "    default:\n" ++
  "      // Fall back to lowest common denominator\n" ++
  s!"      {name}_SM80<<<grid, block, smem, stream>>>({argStr});\n" ++
  "      break;\n" ++
  "  }\n" ++
  "  CUDA_CHECK(cudaGetLastError());\n" ++
  "}\n"

/-- Generate C++ header declarations for all architecture variants -/
def generateHeaderDecls (name : String) (params : Array KParam)
    (archs : Array ArchLevel) : String :=
  let paramDecls := params.toList.map fun p =>
    if p.isPointer then s!"const {p.dtype.toCpp}* {p.name}"
    else s!"{p.dtype.toCpp} {p.name}"
  let paramStr := String.intercalate ", " paramDecls

  let decls := archs.toList.map fun arch =>
    let suffix := arch.toSuffix
    s!"__global__ void {name}{suffix}({paramStr});\n"

  String.join decls ++
  s!"\nvoid {name}_launch(int arch, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, {paramStr});\n"

/-! ## Type-Safe Polymorphic Kernel Builder -/

/-- Builder for polymorphic kernels with type-safe parameters -/
structure PolyKernelBuilder where
  name : String
  params : Array KParam := #[]

namespace PolyKernelBuilder

/-- Add a global pointer parameter -/
def addPtr (b : PolyKernelBuilder) (name : String) (dtype : GpuFloat) : PolyKernelBuilder :=
  { b with params := b.params.push { name, dtype, isPointer := true } }

/-- Add a scalar parameter -/
def addScalar (b : PolyKernelBuilder) (name : String) (dtype : GpuFloat) : PolyKernelBuilder :=
  { b with params := b.params.push { name, dtype, isPointer := false } }

/-- Build with a generator function -/
def build (b : PolyKernelBuilder) (gen : ArchLevel → KernelM Unit) : PolyKernel :=
  mkPolyKernel b.name b.params gen

/-- Build with config access -/
def buildWithConfig (b : PolyKernelBuilder)
    (gen : ArchLevel → ArchCapabilitiesRecord → KernelM Unit) : PolyKernel :=
  mkPolyKernelWithConfig b.name b.params gen

end PolyKernelBuilder

/-- Start building a polymorphic kernel -/
def polyKernel (name : String) : PolyKernelBuilder := { name }

end Tyr.GPU.Codegen.Arch
