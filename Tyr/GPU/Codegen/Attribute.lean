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

/-- The gpu_kernel attribute syntax -/
syntax (name := gpuKernelAttr) "gpu_kernel" term : attr

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

/-- Attribute handler for @[gpu_kernel arch] -/
initialize registerBuiltinAttribute {
  name := `gpuKernelAttr
  descr := "Mark a function as a GPU kernel for the specified architecture"
  applicationTime := .afterCompilation
  add := fun declName stx _attrKind => do
    let `(attr| gpu_kernel $archStx) := stx
      | throwError "Invalid gpu_kernel attribute syntax"

    -- Get the architecture from syntax
    let arch ← match parseArchSyntax archStx with
      | .ok a => pure a
      | .error e => throwError e

    -- Get the declaration info
    let env ← getEnv
    let some info := env.find? declName
      | throwError s!"Declaration {declName} not found"

    -- Extract parameters from the function type
    let params ← Meta.MetaM.run' (extractParams info.type)

    -- Generate companion .kernel definition
    liftCommandElabM (generateKernelCompanion declName arch params)

    -- Register minimal kernel info (full info is in companion definition)
    let registeredKernel : RegisteredKernel := {
      name := declName
      arch := arch
      kernel := { name := declName.toString, arch := arch, params := #[], body := #[] }
      cppCode := s!"// Kernel: {declName}\n// See {declName}.kernel for full definition"
    }

    registerKernel registeredKernel
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
