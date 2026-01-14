/-
  Tyr/GPU/Codegen/FFI.lean

  FFI code generation for GPU kernels.
  Generates Lean @[extern] declarations and C++ launcher wrappers.
-/
import Lean
import Tyr.GPU.Types
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Attribute
import Tyr.GPU.Codegen.EmitNew

namespace Tyr.GPU.Codegen

open Lean Meta Elab Term Command
open Tyr.GPU

/-! ## Lean FFI Declaration Generation -/

/-- Convert kernel name to valid Lean identifier -/
def kernelNameToLeanIdent (name : String) : String :=
  name.replace "." "_"

/-- Convert kernel name to C FFI function name -/
def kernelNameToCFFI (name : String) : String :=
  "lean_launch_" ++ name.replace "." "_"

/-- Generate Lean parameter type for a kernel param -/
def paramToLeanType (p : KParam) : String :=
  if p.isPointer then
    "(@ Tensor)"  -- Borrowed reference to Tensor
  else
    "UInt64"  -- Scalar values as UInt64

/-- Generate Lean @[extern] declaration for a kernel -/
def generateLeanExternDecl (kernel : Kernel) : String :=
  let funcName := s!"launch{kernelNameToLeanIdent kernel.name}"
  let externName := kernelNameToCFFI kernel.name
  let params := kernel.params.toList.map fun p =>
    s!"({p.name} : {paramToLeanType p})"
  let paramStr := String.intercalate " " params
  s!"@[extern \"{externName}\"]\n" ++
  s!"opaque {funcName} {paramStr} (stream : CudaStream) : IO Unit"

/-- Generate Lean launch wrapper with grid/block config -/
def generateLeanLaunchWrapper (kernel : Kernel) : String :=
  let funcName := s!"launch{kernelNameToLeanIdent kernel.name}"
  let externName := s!"{funcName}Impl"
  let params := kernel.params.toList.map fun p =>
    s!"({p.name} : {paramToLeanType p})"
  let paramStr := String.intercalate " " params
  let argNames := kernel.params.toList.map (·.name)
  let argStr := String.intercalate " " argNames

  s!"@[extern \"{kernelNameToCFFI kernel.name}\"]\n" ++
  s!"private opaque {externName} {paramStr} (grid block : UInt64 × UInt64 × UInt64) " ++
  s!"(sharedMem : UInt64) (stream : CudaStream) : IO Unit\n\n" ++
  s!"def {funcName} {paramStr} (grid block : Nat × Nat × Nat := ((1, 1, 1), (128, 1, 1))) " ++
  s!"(sharedMem : Nat := 0) (stream : CudaStream := defaultStream) : IO Unit :=\n" ++
  s!"  let g := (grid.1.toUInt64, grid.2.1.toUInt64, grid.2.2.toUInt64)\n" ++
  s!"  let b := (block.1.toUInt64, block.2.1.toUInt64, block.2.2.toUInt64)\n" ++
  s!"  {externName} {argStr} g b sharedMem.toUInt64 stream"

/-! ## C++ Launcher Generation -/

/-- Convert GpuFloat to C++ CUDA type for pointers -/
def gpuFloatToCudaPtr (dtype : GpuFloat) : String :=
  match dtype with
  | .Float32 => "float*"
  | .Float16 => "__half*"
  | .BFloat16 => "__nv_bfloat16*"
  | .FP8E4M3 => "__nv_fp8_e4m3*"
  | .FP8E5M2 => "__nv_fp8_e5m2*"

/-- Generate C++ extern parameter declaration -/
def paramToCppExternParam (p : KParam) : String :=
  if p.isPointer then
    s!"b_lean_obj_arg {p.name}"
  else
    s!"uint64_t {p.name}"

/-- Generate C++ kernel argument (pointer extraction or scalar pass-through) -/
def paramToCppKernelArg (p : KParam) : String :=
  if p.isPointer then
    s!"{p.name}_ptr"
  else
    p.name

/-- Generate C++ pointer extraction code -/
def generatePointerExtraction (p : KParam) : String :=
  if p.isPointer then
    s!"  auto {p.name}_ptr = borrowTensor({p.name}).data_ptr<{p.dtype.toCpp}>();\n"
  else
    ""

/-- Generate C++ launcher function for a kernel -/
def generateCppLauncher (kernel : Kernel) : String :=
  let externName := kernelNameToCFFI kernel.name
  let externParams := kernel.params.toList.map paramToCppExternParam
  let allParams := externParams ++ [
    "uint64_t grid_x", "uint64_t grid_y", "uint64_t grid_z",
    "uint64_t block_x", "uint64_t block_y", "uint64_t block_z",
    "uint64_t shared_mem", "b_lean_obj_arg stream"
  ]
  let paramStr := String.intercalate ", " allParams

  let ptrExtractions := kernel.params.toList.map generatePointerExtraction
  let extractionCode := String.join ptrExtractions

  let kernelArgs := kernel.params.toList.map paramToCppKernelArg
  let argStr := String.intercalate ", " kernelArgs

  "extern \"C\" lean_object* " ++ externName ++ "(" ++ paramStr ++ ") {\n" ++
  "  try {\n" ++
  extractionCode ++
  "    auto cuda_stream = extractCudaStream(stream);\n" ++
  "    dim3 grid(grid_x, grid_y, grid_z);\n" ++
  "    dim3 block(block_x, block_y, block_z);\n\n" ++
  "    " ++ kernel.name ++ "<<<grid, block, shared_mem, cuda_stream>>>(" ++ argStr ++ ");\n\n" ++
  "    CUDA_CHECK(cudaGetLastError());\n" ++
  "    return lean_io_result_mk_ok(lean_box(0));\n" ++
  "  } catch (const std::exception& e) {\n" ++
  "    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(e.what())));\n" ++
  "  }\n" ++
  "}\n"

/-- Generate complete C++ header for kernel launchers -/
def generateCppHeader : String :=
  "#pragma once\n\n" ++
  "#include <lean/lean.h>\n" ++
  "#include <torch/torch.h>\n" ++
  "#include <cuda_runtime.h>\n" ++
  "#include <cuda_fp16.h>\n" ++
  "#include <cuda_bf16.h>\n\n" ++
  "// Forward declarations\n" ++
  "extern torch::Tensor borrowTensor(b_lean_obj_arg o);\n" ++
  "extern cudaStream_t extractCudaStream(b_lean_obj_arg stream);\n\n" ++
  "#ifndef CUDA_CHECK\n" ++
  "#define CUDA_CHECK(call) do { \\\n" ++
  "  cudaError_t err = call; \\\n" ++
  "  if (err != cudaSuccess) { \\\n" ++
  "    throw std::runtime_error(cudaGetErrorString(err)); \\\n" ++
  "  } \\\n" ++
  "} while(0)\n" ++
  "#endif\n\n"

/-! ## Combined Generation -/

/-- Generate all FFI bindings for a kernel -/
def generateAllBindings (kernel : Kernel) : String × String :=
  let leanCode := generateLeanExternDecl kernel
  let cppCode := generateCppLauncher kernel
  (leanCode, cppCode)

/-- Generate FFI for all registered kernels -/
def generateAllKernelFFI (env : Environment) : String × String :=
  let kernels := getRegisteredKernels env
  let leanDecls := kernels.toList.map fun k => generateLeanExternDecl k.kernel
  let cppLaunchers := kernels.toList.map fun k => generateCppLauncher k.kernel

  let leanCode :=
    "-- Auto-generated kernel FFI declarations\n" ++
    "-- Do not edit manually\n\n" ++
    "namespace Tyr.GPU.Kernels\n\n" ++
    String.intercalate "\n\n" leanDecls ++
    "\n\nend Tyr.GPU.Kernels"

  let cppCode :=
    generateCppHeader ++
    "extern \"C\" {\n\n" ++
    String.intercalate "\n" cppLaunchers ++
    "\n} // extern \"C\"\n"

  (leanCode, cppCode)

/-! ## Commands for FFI Generation -/

/-- Command to generate Lean FFI declarations for all kernels -/
syntax "#generate_lean_ffi" : command

macro_rules
  | `(#generate_lean_ffi) => `(
    #eval do
      let env ← Lean.MonadEnv.getEnv
      let (leanCode, _) := generateAllKernelFFI env
      IO.println leanCode
  )

/-- Command to generate C++ launchers for all kernels -/
syntax "#generate_cpp_ffi" : command

macro_rules
  | `(#generate_cpp_ffi) => `(
    #eval do
      let env ← Lean.MonadEnv.getEnv
      let (_, cppCode) := generateAllKernelFFI env
      IO.println cppCode
  )

/-- Command to write FFI files -/
syntax "#write_kernel_ffi" str str : command

macro_rules
  | `(#write_kernel_ffi $leanPath:str $cppPath:str) => `(
    #eval do
      let env ← Lean.MonadEnv.getEnv
      let (leanCode, cppCode) := generateAllKernelFFI env
      IO.FS.writeFile $leanPath leanCode
      IO.FS.writeFile $cppPath cppCode
      IO.println s!"Written FFI to {$leanPath} and {$cppPath}"
  )

/-! ## Per-Kernel FFI Generation -/

/-- Generate FFI for a specific kernel by name -/
def generateKernelFFI (env : Environment) (name : Name) : Option (String × String) :=
  let kernels := getRegisteredKernels env
  kernels.find? (·.name == name) |>.map fun k =>
    (generateLeanExternDecl k.kernel, generateCppLauncher k.kernel)

/-- Command to print FFI for a specific kernel -/
syntax "#print_kernel_ffi" ident : command

/-! ## Non-Macro FFI Functions

These functions can be called from `#eval` or Lake scripts without needing macros.
-/

/-- Write all registered kernel C++ launchers to a file -/
def writeAllKernelCpp (env : Environment) (path : System.FilePath) : IO Unit := do
  let kernels := getRegisteredKernels env
  let header := generateCppHeader
  let launchers := kernels.toList.map (·.cppCode)
  let cppCode := header ++
    "extern \"C\" {\n\n" ++
    String.intercalate "\n" launchers ++
    "\n} // extern \"C\"\n"
  IO.FS.writeFile path cppCode
  IO.println s!"Written {kernels.size} kernel launchers to {path}"

/-- Get all generated C++ code as a string -/
def getAllKernelCpp (env : Environment) : String :=
  let kernels := getRegisteredKernels env
  let header := generateCppHeader
  let launchers := kernels.toList.map (·.cppCode)
  header ++
    "extern \"C\" {\n\n" ++
    String.intercalate "\n" launchers ++
    "\n} // extern \"C\"\n"

/-- Get all generated Lean FFI declarations as a string -/
def getAllKernelLean (env : Environment) : String :=
  let kernels := getRegisteredKernels env
  let decls := kernels.toList.map fun k => generateLeanExternDecl k.kernel
  "-- Auto-generated kernel FFI declarations\n" ++
  "-- Do not edit manually\n\n" ++
  "namespace Tyr.GPU.Kernels\n\n" ++
  String.intercalate "\n\n" decls ++
  "\n\nend Tyr.GPU.Kernels"

end Tyr.GPU.Codegen
