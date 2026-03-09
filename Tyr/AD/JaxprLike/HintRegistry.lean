import Tyr.AD.JaxprLike.Core
import Tyr.AD.Frontend.Signature

/-!
# Tyr.AD.JaxprLike.HintRegistry

Environment-backed registry for source/front-end lowering hints keyed by
declaration name. This lets higher-level Tyr frontends attach richer metadata
once and reuse the standard `buildFromDecl` / `buildAndExtractFromDecl` path.
-/

namespace Tyr.AD.JaxprLike

structure FnBodyHintRegistry where
  hintsByDecl : Std.HashMap Lean.Name FnBodyLoweringHints := {}
  deriving Inhabited

/--
Frontend-produced AD artifacts for one declaration. This keeps the flat
`LeanJaxpr`, optional structured boundary signature, and optional runtime
frontend constant attached together as one unit at the declaration boundary.
-/
structure FrontendRegistration where
  jaxpr : LeanJaxpr
  signature : Option Tyr.AD.Frontend.FrontendADSignature := none
  /--
  Optional runtime frontend bundle used for generating structured companions
  such as `f.frontend`, `f.linearize`, `f.vjp`, `f.valueAndGrad`, and `f.grad`.
  -/
  runtimeFrontend : Option Lean.Name := none
  deriving Inhabited, Repr

initialize fnBodyHintRegistry : Lean.EnvExtension FnBodyHintRegistry ←
  Lean.registerEnvExtension (pure {})

def registerFnBodyLoweringHints
    (declName : Lean.Name)
    (hints : FnBodyLoweringHints) :
    Lean.CoreM Unit := do
  Lean.modifyEnv fun env =>
    fnBodyHintRegistry.modifyState env fun s =>
      let merged := (s.hintsByDecl.getD declName {}).mergePreferRight hints
      { s with hintsByDecl := s.hintsByDecl.insert declName merged }

def getRegisteredFnBodyLoweringHints
    (declName : Lean.Name) :
    Lean.CoreM FnBodyLoweringHints := do
  return (fnBodyHintRegistry.getState (← Lean.getEnv)).hintsByDecl.getD declName {}

end Tyr.AD.JaxprLike
