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
  /-- Lean compilation unit where this kernel was registered. -/
  moduleName : Name
  /-- Public registration name (used for lookups/debug). -/
  name : Name
  /-- Target architecture -/
  arch : GpuArch
  /-- Kernel definition body (without shared helper templates). -/
  kernelDef : String
  /-- Helper usage flags for reconstructing per-module helper templates. -/
  needsStoreAdd : Bool := false
  needsLegacyTma : Bool := false
  needsSlice : Bool := false
  needsOuter : Bool := false
  /-- Materialized C++ launcher wrapper. -/
  cppCode : String
  /-- Companion constant that can be evaluated later to materialize the IR kernel. -/
  kernelConst : Name
  /-- C++ kernel symbol stem (used by emitted kernel definition + launcher symbol). -/
  kernelName : String
  /-- Extracted launch parameters. -/
  params : Array KParam
  deriving Repr, Inhabited

/-- Lightweight kernel registration stored in the persistent env extension. -/
structure RegisteredKernelSpec where
  /-- Lean compilation unit where this kernel was registered. -/
  moduleName : Name
  /-- Public registration name (used for lookups/debug). -/
  name : Name
  /-- Target architecture -/
  arch : GpuArch
  /-- Companion constant that can be evaluated later to materialize the IR kernel. -/
  kernelConst : Name
  /-- C++ kernel symbol stem (used by emitted kernel definition + launcher symbol). -/
  kernelName : String
  /-- Extracted launch parameters. -/
  params : Array KParam
  deriving Repr, Inhabited

/-- Runtime kernel registry populated by generated module initializers. -/
initialize gpuKernelRegistry : IO.Ref (Array RegisteredKernel) ←
  IO.mkRef #[]

/-- Persistent registry keyed by source declaration name. -/
initialize gpuKernelSpecExt : MapDeclarationExtension (Array RegisteredKernelSpec) ←
  mkMapDeclarationExtension `gpuKernelSpecExt

/-- Marker tag for declarations annotated with `@[gpu_kernel]`. -/
initialize gpuKernelDeclTag : TagAttribute ←
  registerTagAttribute `gpu_kernel_decl
    "Internal marker for declarations annotated with @[gpu_kernel]."

/-- Drop duplicate registrations by companion constant name. -/
private def dedupeKernelSpecs (specs : Array RegisteredKernelSpec) : Array RegisteredKernelSpec := Id.run do
  let mut seen : Std.HashSet Name := {}
  let mut out : Array RegisteredKernelSpec := #[]
  for spec in specs do
    if !seen.contains spec.kernelConst then
      seen := seen.insert spec.kernelConst
      out := out.push spec
  out

/-- Get all kernel specs registered in the environment extension. -/
def getRegisteredKernelSpecs (env : Environment) : Array RegisteredKernelSpec := Id.run do
  let mut specs : Array RegisteredKernelSpec := #[]
  for (_, regs) in gpuKernelSpecExt.getState env |>.toList do
    specs := specs ++ regs
  for modIdx in [:env.header.moduleNames.size] do
    for (_, regs) in gpuKernelSpecExt.getModuleEntries env modIdx do
      specs := specs ++ regs
  dedupeKernelSpecs specs

/-- Register all kernel specs associated with one source declaration. -/
def registerKernelSpecsForDecl (declName : Name) (specs : Array RegisteredKernelSpec) : CoreM Unit := do
  modifyEnv fun env => gpuKernelSpecExt.insert env declName specs

/-- Clear runtime-registered kernels (call before importing kernel modules). -/
def clearRegisteredKernels : IO Unit :=
  gpuKernelRegistry.set #[]

/-- Register one kernel in the runtime registry. -/
def registerKernelRuntime (kernel : RegisteredKernel) : IO Unit :=
  gpuKernelRegistry.modify (·.push kernel)

/-- Replace the runtime kernel registry contents. -/
def setRegisteredKernels (kernels : Array RegisteredKernel) : IO Unit :=
  gpuKernelRegistry.set kernels

/-- Get all kernels currently registered in this process. -/
def getRegisteredKernelsIO : IO (Array RegisteredKernel) :=
  gpuKernelRegistry.get

/-! ## Parameter Extraction -/

/-- Extracted parameter info from function signature -/
structure ExtractedParam where
  name : Name
  dtype : GpuFloat
  isPointer : Bool
  scalarTy : KScalarType := .UInt64
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

/-- Try to extract a supported scalar type from a type expression. -/
def extractScalarType? (type : Expr) : MetaM (Option KScalarType) := do
  let type ← whnf type
  if type.isConstOf ``UInt8 then return some .UInt8
  else if type.isConstOf ``UInt16 then return some .UInt16
  else if type.isConstOf ``UInt32 then return some .UInt32
  else if type.isConstOf ``UInt64 then return some .UInt64
  else if type.isConstOf ``USize then return some .USize
  else if type.isConstOf ``Float then return some .Float
  else if type.isConstOf ``Float32 then return some .Float32
  else if type.isConstOf ``Bool then return some .Bool
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
    let scalarExpr := type.getArg! 0
    let some scalarTy ← extractScalarType? scalarExpr
      | throwError "Unsupported KVal scalar type for parameter '{paramName}'. Use one of: UInt8/16/32/64, USize, Float, Float32, Bool."
    return some { name := paramName, dtype := .Float32, isPointer := false, scalarTy := scalarTy }
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

/-- Convert scalar type to term syntax for `KParam` literals. -/
def scalarTypeToStx (ty : KScalarType) : TSyntax `term :=
  match ty with
  | .UInt8 => mkIdent ``KScalarType.UInt8
  | .UInt16 => mkIdent ``KScalarType.UInt16
  | .UInt32 => mkIdent ``KScalarType.UInt32
  | .UInt64 => mkIdent ``KScalarType.UInt64
  | .USize => mkIdent ``KScalarType.USize
  | .Float => mkIdent ``KScalarType.Float
  | .Float32 => mkIdent ``KScalarType.Float32
  | .Bool => mkIdent ``KScalarType.Bool

/-- Convert scalar type to Lean type syntax for extern declarations. -/
def scalarTypeToLeanTypeStx (ty : KScalarType) : TSyntax `term :=
  match ty with
  | .UInt8 => mkIdent ``UInt8
  | .UInt16 => mkIdent ``UInt16
  | .UInt32 => mkIdent ``UInt32
  | .UInt64 => mkIdent ``UInt64
  | .USize => mkIdent ``USize
  | .Float => mkIdent ``Float
  | .Float32 => mkIdent ``Float32
  | .Bool => mkIdent ``Bool

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
    let scalarTyStx := scalarTypeToStx p.scalarTy
    `({ name := $nameStr, dtype := $dtypeStx, isPointer := $isPtr, scalarTy := $scalarTyStx : KParam })

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

  let nameStr := Syntax.mkStrLit (declName.toString.replace "." "_")

  -- Generate the companion definition.
  let cmd ← `(
    def $(mkIdent companionName) : Kernel :=
      buildKernelM $nameStr $archStx #[$kparamStxs,*] ($fnIdent $argStxs*)
  )

  elabCommand cmd

/-! ## C++ Code Generation -/

/-- Generate C++ extern parameter for a kernel param -/
def paramToCppExternAttr (p : KParam) : String :=
  if p.isPointer then s!"b_lean_obj_arg {p.name}"
  else s!"{p.scalarTy.toCpp} {p.name}"

/-- Generate C++ kernel argument (GL wrapper for pointers, pass-through for scalars). -/
def paramToCppArgAttr (idx : Nat) (p : KParam) : String :=
  if p.isPointer then s!"v{idx}_gl"
  else p.name

/-- Generate pointer extraction code for a param -/
def generatePtrExtractionAttr (idx : Nat) (p : KParam) : String :=
  if p.isPointer then
    s!"    auto v{idx}_tensor = borrowTensor({p.name});\n" ++
    s!"    if (!v{idx}_tensor.is_cuda()) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(\"{p.name} must be a CUDA tensor\")));\n" ++
    s!"    if (!v{idx}_tensor.is_contiguous()) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(\"{p.name} must be contiguous\")));\n" ++
    s!"    if (v{idx}_tensor.dim() > 4) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(\"{p.name} must have dim <= 4\")));\n" ++
    s!"    std::array<int, 4> v{idx}_shape = \{1, 1, 1, 1};\n" ++
    s!"    for (int i = 0; i < static_cast<int>(v{idx}_tensor.dim()); ++i)\n" ++
    s!"      v{idx}_shape[4 - v{idx}_tensor.dim() + i] = static_cast<int>(v{idx}_tensor.size(i));\n" ++
    s!"    using v{idx}_gl_t = gl<{p.dtype.toCpp}, 1, 1, -1, -1>;\n" ++
    s!"    auto v{idx}_gl = kittens::make_gl<v{idx}_gl_t, false>(reinterpret_cast<uint64_t>(v{idx}_tensor.data_ptr()),\n" ++
    s!"      v{idx}_shape[0], v{idx}_shape[1], v{idx}_shape[2], v{idx}_shape[3]);\n"
  else ""

/-- Generate complete C++ launcher code for a kernel -/
def generateCppLauncherCode (kernelName : String) (arch : GpuArch) (params : Array KParam) : String :=
  let name := kernelName.replace "." "_"
  let externName := "lean_launch_" ++ name
  let archGuard := arch.toGuard
  let archMsg := toString arch

  let externParams := params.toList.map paramToCppExternAttr
  let allParams := externParams ++ [
    "uint64_t grid_x", "uint64_t grid_y", "uint64_t grid_z",
    "uint64_t block_x", "uint64_t block_y", "uint64_t block_z",
    "uint64_t shared_mem", "uint64_t stream"
  ]
  let paramStr := String.intercalate ", " allParams

  let extractionCode := Id.run do
    let mut out := ""
    for h : idx in [:params.size] do
      out := out ++ generatePtrExtractionAttr idx params[idx]
    return out

  let kernelArgs := Id.run do
    let mut args : List String := []
    for h : idx in [:params.size] do
      args := args.concat (paramToCppArgAttr idx params[idx])
    return args
  let argStr := String.intercalate ", " kernelArgs

  "extern \"C\" lean_object* " ++ externName ++ "(" ++ paramStr ++ ") {\n" ++
  s!"#if defined({archGuard})\n" ++
  extractionCode ++
  "    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);\n" ++
  "    dim3 grid(grid_x, grid_y, grid_z);\n" ++
  "    dim3 block(block_x, block_y, block_z);\n\n" ++
  "    " ++ name ++ "<<<grid, block, shared_mem, cuda_stream>>>(" ++ argStr ++ ");\n\n" ++
  "    if (auto err = cudaGetLastError(); err != cudaSuccess)\n" ++
  "      return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(cudaGetErrorString(err))));\n" ++
  "    return lean_io_result_mk_ok(lean_box(0));\n" ++
  "#else\n" ++
  s!"    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(\"Kernel {name} is unavailable in this build (requires {archMsg}).\")));\n" ++
  "#endif\n" ++
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
    else do
      let scalarTyStx := scalarTypeToLeanTypeStx p.scalarTy
      `(bracketedBinder| ($paramIdent : $scalarTyStx))
    paramBinders := paramBinders.push binder

  -- Add flattened grid/block dimensions and launch params
  let gridXBinder ← `(bracketedBinder| (grid_x : UInt64))
  let gridYBinder ← `(bracketedBinder| (grid_y : UInt64))
  let gridZBinder ← `(bracketedBinder| (grid_z : UInt64))
  let blockXBinder ← `(bracketedBinder| (block_x : UInt64))
  let blockYBinder ← `(bracketedBinder| (block_y : UInt64))
  let blockZBinder ← `(bracketedBinder| (block_z : UInt64))
  let smemBinder ← `(bracketedBinder| (sharedMem : UInt64))
  let streamBinder ← `(bracketedBinder| (stream : CudaStream))
  paramBinders := paramBinders ++ #[
    gridXBinder, gridYBinder, gridZBinder,
    blockXBinder, blockYBinder, blockZBinder,
    smemBinder, streamBinder
  ]

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

/-- Evaluate an already-defined kernel constant to get the full IR body. -/
unsafe def evalKernelConst (constName : Name) : CoreM Kernel := do
  withTheReader Core.Context (fun ctx => { ctx with maxHeartbeats := 0 }) do
    Meta.MetaM.run' do
      let info ← getConstInfo constName
      let value ← match info with
        | .defnInfo info => pure info.value
        | .thmInfo info => pure info.value
        | _ => throwError "Kernel companion '{constName}' is not reducible."
      evalExpr Kernel (mkConst ``Kernel) value

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
    let scalarTyStx := scalarTypeToStx p.scalarTy
    `({ name := $nameStr, dtype := $dtypeStx, isPointer := $isPtr, scalarTy := $scalarTyStx : KParam })

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

  let polyKernelName := declName.toString.replace "." "_" ++ suffix
  let nameStr := Syntax.mkStrLit polyKernelName

  -- For polymorphic kernels, the function returns ArchKernelM arch Unit
  -- We need to extract the .run to get KernelM Unit
  let cmd <- `(
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
    else do
      let scalarTyStx := scalarTypeToLeanTypeStx p.scalarTy
      `(bracketedBinder| ($paramIdent : $scalarTyStx))
    paramBinders := paramBinders.push binder

  -- Add flattened grid/block dimensions and launch params
  let gridXBinder ← `(bracketedBinder| (grid_x : UInt64))
  let gridYBinder ← `(bracketedBinder| (grid_y : UInt64))
  let gridZBinder ← `(bracketedBinder| (grid_z : UInt64))
  let blockXBinder ← `(bracketedBinder| (block_x : UInt64))
  let blockYBinder ← `(bracketedBinder| (block_y : UInt64))
  let blockZBinder ← `(bracketedBinder| (block_z : UInt64))
  let smemBinder ← `(bracketedBinder| (sharedMem : UInt64))
  let streamBinder ← `(bracketedBinder| (stream : CudaStream))
  paramBinders := paramBinders ++ #[
    gridXBinder, gridYBinder, gridZBinder,
    blockXBinder, blockYBinder, blockZBinder,
    smemBinder, streamBinder
  ]

  -- Build the argument list for each arch-specific launcher
  let mut argIdents : Array (TSyntax `term) := #[]
  for p in params do
    argIdents := argIdents.push (mkIdent p.name)
  argIdents := argIdents ++ #[
    mkIdent `grid_x, mkIdent `grid_y, mkIdent `grid_z,
    mkIdent `block_x, mkIdent `block_y, mkIdent `block_z,
    mkIdent `sharedMem, mkIdent `stream
  ]

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
    else do
      let scalarTyStx := scalarTypeToLeanTypeStx p.scalarTy
      `(bracketedBinder| ($paramIdent : $scalarTyStx))
    paramBinders := paramBinders.push binder

  let gridXBinder ← `(bracketedBinder| (grid_x : UInt64))
  let gridYBinder ← `(bracketedBinder| (grid_y : UInt64))
  let gridZBinder ← `(bracketedBinder| (grid_z : UInt64))
  let blockXBinder ← `(bracketedBinder| (block_x : UInt64))
  let blockYBinder ← `(bracketedBinder| (block_y : UInt64))
  let blockZBinder ← `(bracketedBinder| (block_z : UInt64))
  let smemBinder ← `(bracketedBinder| (sharedMem : UInt64))
  let streamBinder ← `(bracketedBinder| (stream : CudaStream))
  paramBinders := paramBinders ++ #[
    gridXBinder, gridYBinder, gridZBinder,
    blockXBinder, blockYBinder, blockZBinder,
    smemBinder, streamBinder
  ]

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

/-- Run a `CoreM` action in a specific environment. -/
private unsafe def runCoreWithEnv (env : Environment) (x : CoreM α) : IO α := do
  let ctx : Core.Context := {
    fileName := "<gpu-kernel-registry>"
    fileMap := default
  }
  let st : Core.State := { env := env }
  x.toIO' ctx st

/-- Materialize one registered kernel spec into emitted CUDA/C++ fragments. -/
unsafe def materializeRegisteredKernel (spec : RegisteredKernelSpec) : CoreM RegisteredKernel := do
  let kernel ← evalKernelConst spec.kernelConst
  let emitInfo := generateKernelEmitInfo kernel
  let cppCode := generateCppLauncherCode spec.kernelName spec.arch spec.params
  return {
    moduleName := spec.moduleName
    name := spec.name
    arch := spec.arch
    kernelDef := emitInfo.definition
    needsStoreAdd := emitInfo.needsStoreAdd
    needsLegacyTma := emitInfo.needsLegacyTma
    needsSlice := emitInfo.needsSlice
    needsOuter := emitInfo.needsOuter
    cppCode := cppCode
    kernelConst := spec.kernelConst
    kernelName := spec.kernelName
    params := spec.params
  }

/-- Collect all registered kernels from an imported environment. -/
unsafe def collectRegisteredKernelsFromEnv (env : Environment) : IO (Array RegisteredKernel) := do
  runCoreWithEnv env do
    let specs := getRegisteredKernelSpecs (← getEnv)
    specs.mapM materializeRegisteredKernel

/-- Collect registered kernels only from the requested Lean modules. -/
unsafe def collectRegisteredKernelsFromEnvModules
    (env : Environment) (modules : Array Name) : IO (Array RegisteredKernel) := do
  runCoreWithEnv env do
    let specs := getRegisteredKernelSpecs (← getEnv)
    let selected :=
      if modules.isEmpty then
        specs
      else
        specs.filter (fun s => modules.contains s.moduleName)
    selected.mapM materializeRegisteredKernel

/-- Attribute handler for @[gpu_kernel] and @[gpu_kernel arch] -/
initialize registerBuiltinAttribute {
  name := `gpuKernelAttr
  descr := "Mark a function as a GPU kernel. Use @[gpu_kernel .SM90] for single arch, @[gpu_kernel] for polymorphic."
  applicationTime := .afterTypeChecking
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
      let moduleName := env.mainModule
      let arch ← match parseArchSyntax archStx with
        | .ok a => pure a
        | .error e => throwError e

      liftCommandElabM (generateKernelCompanion declName arch params)
      liftCommandElabM (generateLaunchDecl declName params)
      gpuKernelDeclTag.setTag declName

      let kparams := params.map fun p =>
        { name := p.name.toString, dtype := p.dtype, isPointer := p.isPointer, scalarTy := p.scalarTy : KParam }
      let kernelConst := declName ++ `kernel
      let kernelName := declName.toString.replace "." "_"

      let regSpec : RegisteredKernelSpec := {
        moduleName := moduleName
        name := declName
        arch := arch
        kernelConst := kernelConst
        kernelName := kernelName
        params := kparams
      }
      registerKernelSpecsForDecl declName #[regSpec]

    | `(attr| gpu_kernel) =>
      -- Polymorphic mode: check if function has ArchLevel parameter
      let moduleName := env.mainModule
      let isPoly ← Meta.MetaM.run' (hasArchLevelParam info.type)
      if !isPoly then
        throwError "Polymorphic @[gpu_kernel] requires function to have an ArchLevel parameter"

      -- Generate kernel for each architecture
      let archs := #[
        (GpuArch.SM80, Arch.ArchLevel.Ampere),
        (GpuArch.SM90, Arch.ArchLevel.Hopper),
        (GpuArch.SM100, Arch.ArchLevel.Blackwell)
      ]

      let mut regSpecs : Array RegisteredKernelSpec := #[]
      for (gpuArch, archLevel) in archs do
        liftCommandElabM (generatePolyKernelCompanion declName gpuArch archLevel params)
        liftCommandElabM (generateArchLaunchDecl declName archLevel.toSuffix params)

        let kparams := params.map fun p =>
          { name := p.name.toString, dtype := p.dtype, isPointer := p.isPointer, scalarTy := p.scalarTy : KParam }
        let kernelCompanion := declName ++ Name.mkSimple s!"kernel{archLevel.toSuffix}"
        let kernelName := declName.toString.replace "." "_" ++ archLevel.toSuffix

        let regSpec : RegisteredKernelSpec := {
          moduleName := moduleName
          name := declName ++ archLevel.toNameSuffix
          arch := gpuArch
          kernelConst := kernelCompanion
          kernelName := kernelName
          params := kparams
        }
        regSpecs := regSpecs.push regSpec

      registerKernelSpecsForDecl declName regSpecs

      -- Generate unified launcher
      liftCommandElabM (generatePolyLaunchDecl declName params)
      gpuKernelDeclTag.setTag declName

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
      let kernels ← (unsafe collectRegisteredKernelsFromEnv env)
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
