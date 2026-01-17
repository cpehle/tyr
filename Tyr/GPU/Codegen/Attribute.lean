/-
  Tyr/GPU/Codegen/Attribute.lean

  @[gpu_kernel] attribute for marking GPU kernel functions.
  Provides automatic parameter extraction and kernel registration.
-/
import Lean
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.Arch.Level

namespace Tyr.GPU.Codegen

open Lean Meta Elab Term Command
open Tyr.GPU

/-! ## Kernel Registry -/

/-- Registered GPU kernel -/
structure RegisteredKernel where
  /-- Kernel name -/
  name : Name
  /-- Target architecture -/
  arch : GpuArch
  /-- The compiled kernel -/
  kernel : Kernel
  /-- Generated C++ code -/
  cppCode : String
  deriving Repr, Inhabited

/-- Global kernel registry (stored in environment extension) -/
initialize gpuKernelExt : SimplePersistentEnvExtension RegisteredKernel (Array RegisteredKernel) ←
  registerSimplePersistentEnvExtension {
    addEntryFn := fun arr k => arr.push k
    addImportedFn := fun arrs => arrs.foldl (· ++ ·) #[]
  }

/-- Get all registered kernels -/
def getRegisteredKernels (env : Environment) : Array RegisteredKernel :=
  gpuKernelExt.getState env

/-- Register a kernel -/
def registerKernel (kernel : RegisteredKernel) : CoreM Unit := do
  modifyEnv fun env => gpuKernelExt.addEntry env kernel

/-! ## Parameter Extraction -/

/-- Extracted parameter info from function signature -/
structure ExtractedParam where
  name : Name
  dtype : GpuFloat
  isPointer : Bool
  deriving Repr, Inhabited

/-- Try to extract GpuFloat from a type expression -/
def extractGpuFloat? (type : Expr) : MetaM (Option GpuFloat) := do
  let type ← whnf type
  if type.isConstOf ``GpuFloat.Float32 then return some .Float32
  else if type.isConstOf ``GpuFloat.Float16 then return some .Float16
  else if type.isConstOf ``GpuFloat.BFloat16 then return some .BFloat16
  else if type.isConstOf ``GpuFloat.FP8E4M3 then return some .FP8E4M3
  else if type.isConstOf ``GpuFloat.FP8E5M2 then return some .FP8E5M2
  else return none

/-- Try to extract parameter info from GPtr or KVal types -/
def extractParamFromType? (paramName : Name) (type : Expr) : MetaM (Option ExtractedParam) := do
  let type ← whnf type
  -- Check if it's GPtr dtype
  if type.isAppOfArity ``GPtr 1 then
    let dtypeExpr := type.getArg! 0
    if let some dtype ← extractGpuFloat? dtypeExpr then
      return some { name := paramName, dtype := dtype, isPointer := true }
  -- Check if it's KVal T (maps to scalar)
  if type.isAppOfArity ``KVal 1 then
    -- KVal maps to Float32 scalar for now
    return some { name := paramName, dtype := .Float32, isPointer := false }
  return none

/-- Extract parameters from function type -/
partial def extractParams (type : Expr) : MetaM (Array ExtractedParam) := do
  let mut params := #[]
  let mut currType := type

  while currType.isForall do
    let name := currType.bindingName!
    let domainType := currType.bindingDomain!

    if let some param ← extractParamFromType? name domainType then
      params := params.push param

    currType ← whnf (currType.bindingBody!.instantiate1 (mkFVar ⟨name⟩))

  return params

/-! ## Kernel Attribute Implementation -/

/-- Parse architecture from syntax -/
def parseArch (stx : Syntax) : MetaM GpuArch := do
  match stx with
  | `(.SM80) => return .SM80
  | `(.SM90) => return .SM90
  | `(.SM100) => return .SM100
  | _ => throwError "Invalid GPU architecture: expected .SM80, .SM90, or .SM100"

/-- The gpu_kernel attribute syntax
    - @[gpu_kernel .SM90] - single architecture kernel
    - @[gpu_kernel] - polymorphic kernel (generates for all architectures) -/
syntax (name := gpuKernelAttr) "gpu_kernel" (term)? : attr

/-- Check if a substring exists in a string -/
def containsSubstr (s sub : String) : Bool :=
  (s.splitOn sub).length > 1

/-- Parse architecture from a syntax node -/
def parseArchSyntax (stx : Syntax) : Except String GpuArch :=
  let s := stx.reprint.getD ""
  if containsSubstr s "SM80" then .ok .SM80
  else if containsSubstr s "SM90" then .ok .SM90
  else if containsSubstr s "SM100" then .ok .SM100
  else .error s!"Invalid architecture: {s}"

/-- Generate companion kernel definition as syntax -/
def generateKernelCompanion (declName : Name) (arch : GpuArch)
    (params : Array ExtractedParam) : CommandElabM Unit := do
  let companionName := declName ++ `kernel
  let fnIdent := mkIdent declName

  -- Build KParam array syntax
  let kparamStxs ← params.mapM fun p => do
    let dtypeStx := match p.dtype with
      | .Float32 => mkIdent ``GpuFloat.Float32
      | .Float16 => mkIdent ``GpuFloat.Float16
      | .BFloat16 => mkIdent ``GpuFloat.BFloat16
      | .FP8E4M3 => mkIdent ``GpuFloat.FP8E4M3
      | .FP8E5M2 => mkIdent ``GpuFloat.FP8E5M2
    let nameStr := Syntax.mkStrLit p.name.toString
    let isPtr := if p.isPointer then mkIdent ``true else mkIdent ``false
    `({ name := $nameStr, dtype := $dtypeStx, isPointer := $isPtr : KParam })

  -- Build argument syntax for each parameter
  let mut argStxs : Array (TSyntax `term) := #[]
  for h : idx in [:params.size] do
    let p := params[idx]
    let idxLit := Syntax.mkNumLit (toString idx)
    let nameStr := Syntax.mkStrLit p.name.toString
    let argStx : TSyntax `term ← if p.isPointer then
      `(GPtr.mk ⟨$idxLit⟩ $nameStr)
    else
      `(KVal.mk ⟨$idxLit⟩ $nameStr)
    argStxs := argStxs.push argStx

  -- Build arch syntax
  let archStx := match arch with
    | .SM80 => mkIdent ``GpuArch.SM80
    | .SM90 => mkIdent ``GpuArch.SM90
    | .SM100 => mkIdent ``GpuArch.SM100

  let nameStr := Syntax.mkStrLit declName.toString

  -- Generate the definition and ensure it's compiled
  let cmd ← `(
    def $(mkIdent companionName) : Kernel :=
      buildKernelM $nameStr $archStx #[$kparamStxs,*] ($fnIdent $argStxs*)
  )

  elabCommand cmd

/-! ## C++ Code Generation -/

/-- Generate C++ extern parameter for a kernel param -/
def paramToCppExternAttr (p : KParam) : String :=
  if p.isPointer then s!"b_lean_obj_arg {p.name}"
  else s!"uint64_t {p.name}"

/-- Generate C++ kernel argument (pointer extraction or pass-through) -/
def paramToCppArgAttr (p : KParam) : String :=
  if p.isPointer then s!"{p.name}_ptr"
  else p.name

/-- Generate pointer extraction code for a param -/
def generatePtrExtractionAttr (p : KParam) : String :=
  if p.isPointer then
    s!"    auto {p.name}_ptr = borrowTensor({p.name}).data_ptr<{p.dtype.toCpp}>();\n"
  else ""

/-- Generate complete C++ launcher code for a kernel -/
def generateCppLauncherCode (declName : Name) (params : Array KParam) : String :=
  let name := declName.toString.replace "." "_"
  let externName := "lean_launch_" ++ name

  let externParams := params.toList.map paramToCppExternAttr
  let allParams := externParams ++ [
    "uint64_t grid_x", "uint64_t grid_y", "uint64_t grid_z",
    "uint64_t block_x", "uint64_t block_y", "uint64_t block_z",
    "uint64_t shared_mem", "b_lean_obj_arg stream"
  ]
  let paramStr := String.intercalate ", " allParams

  let ptrExtractions := params.toList.map generatePtrExtractionAttr
  let extractionCode := String.join ptrExtractions

  let kernelArgs := params.toList.map paramToCppArgAttr
  let argStr := String.intercalate ", " kernelArgs

  "extern \"C\" lean_object* " ++ externName ++ "(" ++ paramStr ++ ") {\n" ++
  "  try {\n" ++
  extractionCode ++
  "    auto cuda_stream = extractCudaStream(stream);\n" ++
  "    dim3 grid(grid_x, grid_y, grid_z);\n" ++
  "    dim3 block(block_x, block_y, block_z);\n\n" ++
  "    " ++ name ++ "<<<grid, block, shared_mem, cuda_stream>>>(" ++ argStr ++ ");\n\n" ++
  "    CUDA_CHECK(cudaGetLastError());\n" ++
  "    return lean_io_result_mk_ok(lean_box(0));\n" ++
  "  } catch (const std::exception& e) {\n" ++
  "    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(e.what())));\n" ++
  "  }\n" ++
  "}\n"

/-- Generate the FFI launch declaration for a kernel
    Creates: @[extern "lean_launch_X"] opaque X.launch (params...) : IO Unit -/
def generateLaunchDecl (declName : Name) (params : Array ExtractedParam) : CommandElabM Unit := do
  let launchName := declName ++ `launch
  let externNameStr := "lean_launch_" ++ declName.toString.replace "." "_"

  -- Build parameter binders: (a : @& Tensor) (b : @& Tensor) (size : UInt64)
  let mut paramBinders : Array (TSyntax `Lean.Parser.Term.bracketedBinder) := #[]
  for p in params do
    let paramIdent := mkIdent p.name
    let binder ← if p.isPointer then
      `(bracketedBinder| ($paramIdent : @& Tensor))
    else
      `(bracketedBinder| ($paramIdent : UInt64))
    paramBinders := paramBinders.push binder

  -- Add grid, block, sharedMem, stream parameters
  let gridBinder ← `(bracketedBinder| (grid : UInt64 × UInt64 × UInt64))
  let blockBinder ← `(bracketedBinder| (block : UInt64 × UInt64 × UInt64))
  let smemBinder ← `(bracketedBinder| (sharedMem : UInt64))
  let streamBinder ← `(bracketedBinder| (stream : CudaStream))
  paramBinders := paramBinders ++ #[gridBinder, blockBinder, smemBinder, streamBinder]

  -- Build extern entry syntax node:
  -- externEntry := optional (ident >> ppSpace) >> optional (nonReservedSymbol "inline ") >> strLit
  -- For simple case: just the string literal with nulls for optional parts
  let strLitNode := Syntax.mkStrLit externNameStr
  let externEntryStx : TSyntax `Lean.Parser.Attr.externEntry := ⟨Syntax.node SourceInfo.none
    `Lean.Parser.Attr.externEntry #[
      Syntax.node SourceInfo.none `null #[],  -- optional ident (empty)
      Syntax.node SourceInfo.none `null #[],  -- optional "inline" (empty)
      strLitNode                               -- strLit
    ]⟩

  -- Generate: @[extern "lean_launch_X"] opaque X.launch (params...) : IO Unit
  let cmd ← `(command|
    @[extern $externEntryStx]
    opaque $(mkIdent launchName) $paramBinders* : IO Unit
  )

  elabCommand cmd

/-- Check if a function type has an ArchLevel parameter -/
def hasArchLevelParam (type : Expr) : MetaM Bool := do
  let mut currType := type
  while currType.isForall do
    let domainType ← whnf currType.bindingDomain!
    if domainType.isConstOf ``Arch.ArchLevel then
      return true
    currType ← whnf (currType.bindingBody!.instantiate1 (mkFVar ⟨currType.bindingName!⟩))
  return false

/-- Generate companion kernel definition for a polymorphic kernel at a specific architecture -/
def generatePolyKernelCompanion (declName : Name) (arch : GpuArch) (archLevel : Arch.ArchLevel)
    (params : Array ExtractedParam) : CommandElabM Unit := do
  let suffix := archLevel.toSuffix
  let companionName := declName ++ Name.mkSimple s!"kernel{suffix}"
  let fnIdent := mkIdent declName

  -- Build KParam array syntax
  let kparamStxs ← params.mapM fun p => do
    let dtypeStx := match p.dtype with
      | .Float32 => mkIdent ``GpuFloat.Float32
      | .Float16 => mkIdent ``GpuFloat.Float16
      | .BFloat16 => mkIdent ``GpuFloat.BFloat16
      | .FP8E4M3 => mkIdent ``GpuFloat.FP8E4M3
      | .FP8E5M2 => mkIdent ``GpuFloat.FP8E5M2
    let nameStr := Syntax.mkStrLit p.name.toString
    let isPtr := if p.isPointer then mkIdent ``true else mkIdent ``false
    `({ name := $nameStr, dtype := $dtypeStx, isPointer := $isPtr : KParam })

  -- Build argument syntax for each parameter
  let mut argStxs : Array (TSyntax `term) := #[]
  for h : idx in [:params.size] do
    let p := params[idx]
    let idxLit := Syntax.mkNumLit (toString idx)
    let nameStr := Syntax.mkStrLit p.name.toString
    let argStx : TSyntax `term ← if p.isPointer then
      `(GPtr.mk ⟨$idxLit⟩ $nameStr)
    else
      `(KVal.mk ⟨$idxLit⟩ $nameStr)
    argStxs := argStxs.push argStx

  -- Add the architecture argument
  let archLevelStx := match archLevel with
    | .Ampere => mkIdent ``Arch.ArchLevel.Ampere
    | .Hopper => mkIdent ``Arch.ArchLevel.Hopper
    | .Blackwell => mkIdent ``Arch.ArchLevel.Blackwell
  argStxs := argStxs.push archLevelStx

  let archStx := match arch with
    | .SM80 => mkIdent ``GpuArch.SM80
    | .SM90 => mkIdent ``GpuArch.SM90
    | .SM100 => mkIdent ``GpuArch.SM100

  let nameStr := Syntax.mkStrLit s!"{declName}{suffix}"

  -- For polymorphic kernels, the function returns ArchKernelM arch Unit
  -- We need to extract the .run to get KernelM Unit
  let cmd ← `(
    def $(mkIdent companionName) : Kernel :=
      buildKernelM $nameStr $archStx #[$kparamStxs,*] (($fnIdent $argStxs*).run)
  )

  elabCommand cmd

/-- Generate unified launch declaration for polymorphic kernel -/
def generatePolyLaunchDecl (declName : Name) (params : Array ExtractedParam) : CommandElabM Unit := do
  let launchName := declName ++ `launch

  -- Build parameter binders
  let mut paramBinders : Array (TSyntax `Lean.Parser.Term.bracketedBinder) := #[]

  -- Add arch parameter first
  let archBinder ← `(bracketedBinder| (arch : Arch.ArchLevel))
  paramBinders := paramBinders.push archBinder

  for p in params do
    let paramIdent := mkIdent p.name
    let binder ← if p.isPointer then
      `(bracketedBinder| ($paramIdent : @& Tensor))
    else
      `(bracketedBinder| ($paramIdent : UInt64))
    paramBinders := paramBinders.push binder

  -- Add grid, block, sharedMem, stream parameters
  let gridBinder ← `(bracketedBinder| (grid : UInt64 × UInt64 × UInt64))
  let blockBinder ← `(bracketedBinder| (block : UInt64 × UInt64 × UInt64))
  let smemBinder ← `(bracketedBinder| (sharedMem : UInt64))
  let streamBinder ← `(bracketedBinder| (stream : CudaStream))
  paramBinders := paramBinders ++ #[gridBinder, blockBinder, smemBinder, streamBinder]

  -- Build the argument list for each arch-specific launcher
  let mut argIdents : Array (TSyntax `term) := #[]
  for p in params do
    argIdents := argIdents.push (mkIdent p.name)
  argIdents := argIdents ++ #[mkIdent `grid, mkIdent `block, mkIdent `sharedMem, mkIdent `stream]

  let launchSM80 := mkIdent (declName ++ Name.mkSimple "launch_SM80")
  let launchSM90 := mkIdent (declName ++ Name.mkSimple "launch_SM90")
  let launchSM100 := mkIdent (declName ++ Name.mkSimple "launch_SM100")

  -- Generate: def X.launch (arch : ArchLevel) (params...) : IO Unit := match arch with ...
  let cmd ← `(command|
    def $(mkIdent launchName) $paramBinders* : IO Unit :=
      match arch with
      | .Ampere => $launchSM80 $argIdents*
      | .Hopper => $launchSM90 $argIdents*
      | .Blackwell => $launchSM100 $argIdents*
  )

  elabCommand cmd

/-- Generate arch-specific launch declaration -/
def generateArchLaunchDecl (declName : Name) (suffix : String) (params : Array ExtractedParam)
    : CommandElabM Unit := do
  let launchName := declName ++ Name.mkSimple s!"launch{suffix}"
  let externNameStr := "lean_launch_" ++ declName.toString.replace "." "_" ++ suffix

  -- Build parameter binders
  let mut paramBinders : Array (TSyntax `Lean.Parser.Term.bracketedBinder) := #[]
  for p in params do
    let paramIdent := mkIdent p.name
    let binder ← if p.isPointer then
      `(bracketedBinder| ($paramIdent : @& Tensor))
    else
      `(bracketedBinder| ($paramIdent : UInt64))
    paramBinders := paramBinders.push binder

  let gridBinder ← `(bracketedBinder| (grid : UInt64 × UInt64 × UInt64))
  let blockBinder ← `(bracketedBinder| (block : UInt64 × UInt64 × UInt64))
  let smemBinder ← `(bracketedBinder| (sharedMem : UInt64))
  let streamBinder ← `(bracketedBinder| (stream : CudaStream))
  paramBinders := paramBinders ++ #[gridBinder, blockBinder, smemBinder, streamBinder]

  let strLitNode := Syntax.mkStrLit externNameStr
  let externEntryStx : TSyntax `Lean.Parser.Attr.externEntry := ⟨Syntax.node SourceInfo.none
    `Lean.Parser.Attr.externEntry #[
      Syntax.node SourceInfo.none `null #[],
      Syntax.node SourceInfo.none `null #[],
      strLitNode
    ]⟩

  let cmd ← `(command|
    @[extern $externEntryStx]
    opaque $(mkIdent launchName) $paramBinders* : IO Unit
  )

  elabCommand cmd

/-- Attribute handler for @[gpu_kernel] and @[gpu_kernel arch] -/
initialize registerBuiltinAttribute {
  name := `gpuKernelAttr
  descr := "Mark a function as a GPU kernel. Use @[gpu_kernel .SM90] for single arch, @[gpu_kernel] for polymorphic."
  applicationTime := .afterCompilation
  add := fun declName stx _attrKind => do
    -- Get the declaration info
    let env ← getEnv
    let some info := env.find? declName
      | throwError s!"Declaration {declName} not found"

    -- Extract parameters from the function type
    let params ← Meta.MetaM.run' (extractParams info.type)

    -- Check if architecture is specified
    match stx with
    | `(attr| gpu_kernel $archStx) =>
      -- Single architecture mode (existing behavior)
      let arch ← match parseArchSyntax archStx with
        | .ok a => pure a
        | .error e => throwError e

      liftCommandElabM (generateKernelCompanion declName arch params)
      liftCommandElabM (generateLaunchDecl declName params)

      let kparams := params.map fun p =>
        { name := p.name.toString, dtype := p.dtype, isPointer := p.isPointer : KParam }
      let cppCode := generateCppLauncherCode declName kparams

      registerKernel {
        name := declName
        arch := arch
        kernel := { name := declName.toString, arch := arch, params := kparams, body := #[] }
        cppCode := cppCode
      }

    | `(attr| gpu_kernel) =>
      -- Polymorphic mode: check if function has ArchLevel parameter
      let isPoly ← Meta.MetaM.run' (hasArchLevelParam info.type)
      if !isPoly then
        throwError "Polymorphic @[gpu_kernel] requires function to have an ArchLevel parameter"

      -- Generate kernel for each architecture
      let archs := #[
        (GpuArch.SM80, Arch.ArchLevel.Ampere),
        (GpuArch.SM90, Arch.ArchLevel.Hopper),
        (GpuArch.SM100, Arch.ArchLevel.Blackwell)
      ]

      for (gpuArch, archLevel) in archs do
        liftCommandElabM (generatePolyKernelCompanion declName gpuArch archLevel params)
        liftCommandElabM (generateArchLaunchDecl declName archLevel.toSuffix params)

        let kparams := params.map fun p =>
          { name := p.name.toString, dtype := p.dtype, isPointer := p.isPointer : KParam }
        let cppCode := generateCppLauncherCode (declName ++ Name.mkSimple archLevel.toSuffix) kparams

        registerKernel {
          name := declName ++ archLevel.toNameSuffix
          arch := gpuArch
          kernel := { name := s!"{declName}{archLevel.toSuffix}", arch := gpuArch, params := kparams, body := #[] }
          cppCode := cppCode
        }

      -- Generate unified launcher
      liftCommandElabM (generatePolyLaunchDecl declName params)

    | _ => throwError "Invalid gpu_kernel attribute syntax"
}

/-! ## Kernel Building Macro -/

/-- Build a kernel from a KernelM function with explicit parameters -/
def buildGpuKernel (name : String) (arch : GpuArch) (params : Array KParam)
    (body : KernelM Unit) : Kernel :=
  buildKernelM name arch params body

/-- Macro to define and register a GPU kernel in one step -/
macro "gpu_kernel" name:ident arch:term params:term ":=" body:term : command => `(
  def $name : Kernel := buildGpuKernel $(quote name.getId.toString) $arch $params $body
)

/-! ## Code Generation Commands -/

/-- Command to generate C++ for all registered kernels -/
syntax "#generate_gpu_kernels" : command

macro_rules
  | `(#generate_gpu_kernels) => `(
    #eval do
      let env ← Lean.MonadEnv.getEnv
      let kernels := getRegisteredKernels env
      for k in kernels do
        IO.println s!"=== {k.name} ({k.arch}) ==="
        IO.println k.cppCode
  )

/-- Command to print a specific kernel's C++ code -/
syntax "#print_gpu_kernel" ident : command

/-! ## Helper for defining kernels with the new syntax -/

/-- Define a kernel function that returns KernelM Unit -/
abbrev GpuKernelFn := KernelM Unit

/-- Wrapper to make kernel definitions cleaner -/
def kernel (arch : GpuArch) (body : KernelM Unit) : KernelM Unit := do
  setArch arch
  body

end Tyr.GPU.Codegen
