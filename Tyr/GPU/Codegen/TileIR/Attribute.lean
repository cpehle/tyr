import Lean
import Tyr.GPU.Codegen.TileIR.Render

/-!
# Tyr.GPU.Codegen.TileIR.Attribute

Minimal attribute-based registration for Lean TileIR module declarations.

The current backend-first goal is simpler than the Python DSL replacement:

- author TileIR modules directly in Lean data,
- mark them with `@[tileir_kernel]`,
- and generate a rendered-MLIR companion definition for each declaration.

That gives us an attribute-annotated entrypoint now, without committing yet to a
full quoted syntax or frontend elaborator.
-/

namespace Tyr.GPU.Codegen.TileIR

open Lean Meta Elab Command

structure RegisteredTileKernel where
  moduleName : Name
  declName : Name
  mlirCompanion? : Option Name := none
  constParamCount : Nat := 0
  deriving Repr, Inhabited

initialize tileKernelExt : MapDeclarationExtension RegisteredTileKernel ←
  mkMapDeclarationExtension `tileKernelExt

initialize tileKernelDeclTag : TagAttribute ←
  registerTagAttribute `tileir_kernel_decl
    "Internal marker for declarations annotated with @[tileir_kernel]."

syntax (name := tileirKernelAttr) "tileir_kernel" : attr

private partial def tileIRConstParamCount? (ty : Expr) : MetaM (Option Nat) := do
  let ty ← whnf ty
  if ty.isConstOf ``Tyr.GPU.Codegen.TileIR.Module then
    pure (some 0)
  else if ty.isForall then
    let body := ty.bindingBody!.instantiate1 (mkFVar ⟨ty.bindingName!⟩)
    match ← tileIRConstParamCount? body with
    | some count => pure (some (count + 1))
    | none => pure none
  else
    pure none

def mlirCompanionName (declName : Name) : Name :=
  declName ++ `mlir

private def generateMlirCompanion (declName : Name) : CommandElabM Name := do
  let companionName := mlirCompanionName declName
  let cmd ← `(
    abbrev $(mkIdent companionName) : String :=
      Tyr.GPU.Codegen.TileIR.renderOptimizedModule $(mkIdent declName)
  )
  elabCommand cmd
  pure companionName

def getRegisteredTileKernels (env : Environment) : Array RegisteredTileKernel := Id.run do
  let mut kernels : Array RegisteredTileKernel := #[]
  for (declName, _) in env.constants.map₁.toList do
    if let some kernel := tileKernelExt.find? env declName then
      kernels := kernels.push kernel
  kernels

def collectRegisteredTileKernelsFromModules
    (env : Environment)
    (modules : Array Name)
    : Array RegisteredTileKernel :=
  (getRegisteredTileKernels env).filter fun kernel =>
    modules.contains kernel.moduleName

initialize registerBuiltinAttribute {
  name := `tileirKernelAttr
  descr := "Register a Lean declaration as a TileIR module and generate an MLIR companion."
  applicationTime := .afterTypeChecking
  add := fun declName stx _attrKind => do
    if stx.isMissing || stx.isOfKind ``tileirKernelAttr then
        let env ← getEnv
        let some info := env.find? declName
          | throwError s!"Declaration {declName} not found"
        let some constParamCount ← Meta.MetaM.run' (tileIRConstParamCount? info.type)
          | throwError "The @[tileir_kernel] attribute only supports declarations whose result type is `Tyr.GPU.Codegen.TileIR.Module`."
        let moduleName := env.mainModule
        let mlirCompanion? := none
        modifyEnv fun env =>
          tileKernelExt.insert env declName {
            moduleName := moduleName
            declName := declName
            mlirCompanion? := mlirCompanion?
            constParamCount := constParamCount
          }
        tileKernelDeclTag.setTag declName
    else
      throwError "invalid tileir_kernel attribute syntax"
}

end Tyr.GPU.Codegen.TileIR
