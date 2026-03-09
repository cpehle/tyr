import Tyr.AD.JaxprLike.LowerFnBody
import Tyr.AD.JaxprLike.LowerKStmt
import Tyr.AD.JaxprLike.Validate
import Tyr.AD.JaxprLike.Elab
import Tyr.AD.JaxprLike.RuleCheck
import Tyr.AD.JaxprLike.Extract
import Tyr.AD.Frontend.Signature

/-!
# Tyr.AD.JaxprLike.Pipeline

Unified build pipeline:
- convert source IR to LeanJaxpr
- validate structural invariants
- optionally enforce strict local-Jacobian rule coverage
-/

namespace Tyr.AD.JaxprLike

open Lean
open Lean.IR

structure BuildConfig where
  requireRuleCoverage : Bool := true
  deriving Inhabited

inductive BuildError where
  | conversion (message : String)
  | validation (messages : Array String)
  | coverage (messages : Array String)
  deriving Repr, Inhabited

inductive BuildExtractError where
  | build (err : BuildError)
  | ruleExecution (messages : Array String)
  deriving Repr, Inhabited

def buildErrorToString : BuildError → String
  | .conversion msg => s!"Conversion error: {msg}"
  | .validation msgs =>
    "Validation error(s):\n" ++ String.intercalate "\n" msgs.toList
  | .coverage msgs =>
    "Rule coverage error(s):\n" ++ String.intercalate "\n" msgs.toList

def buildExtractErrorToString : BuildExtractError → String
  | .build err => buildErrorToString err
  | .ruleExecution msgs =>
    "Rule execution error(s):\n" ++ String.intercalate "\n" msgs.toList

private def runValidation (jaxpr : LeanJaxpr) : Except BuildError Unit :=
  match validate jaxpr with
  | .ok () => .ok ()
  | .error errs => .error (.validation errs)

private def runCoverage
    (cfg : BuildConfig)
    (jaxpr : LeanJaxpr) :
    CoreM (Except BuildError Unit) := do
  if !cfg.requireRuleCoverage then
    return .ok ()
  else
    match (← requireFullRuleCoverage jaxpr) with
    | .ok () => return .ok ()
    | .error errs =>
      let msgs := errs.map coverageErrorToString
      return .error (.coverage msgs)

private def declNameOf : Decl → Name
  | .fdecl f _ _ _ _ => f
  | .extern f _ _ _ => f

/--
Convert + validate + optionally rule-check from a Lean IR declaration.

When a frontend has already registered a direct `LeanJaxpr` for the declaration,
that high-level IR is preferred over `FnBody` recovery.
-/
def buildFromDecl
    (cfg : BuildConfig := {})
    (decl : Decl) :
    CoreM (Except BuildError LeanJaxpr) := do
  let declName := declNameOf decl
  let frontend? ← getRegisteredFrontendRegistration? declName
  let jaxpr ←
    match frontend? with
    | some reg => pure (reg.jaxpr.withDefaultSourceDecl declName)
    | none =>
      match (← LowerFnBody.runDecl { decl := decl }) with
      | .ok j => pure j
      | .error err => return .error (.conversion (toString err))

  match frontend?.bind (·.signature) with
  | some sig =>
      match sig.validateJaxprBoundaryMetadata jaxpr with
      | .ok () => pure ()
      | .error err => return .error (.conversion err)
  | none => pure ()

  match runValidation jaxpr with
  | .error err => return .error err
  | .ok () =>
    match (← runCoverage cfg jaxpr) with
    | .error err => return .error err
    | .ok () => return .ok jaxpr

/-- Convert + validate + optionally rule-check from raw `FnBody`. -/
def buildFromFnBody
    (cfg : BuildConfig := {})
    (declName : Name)
    (params : Array Param)
    (body : FnBody)
    (hints : FnBodyLoweringHints := {}) :
    CoreM (Except BuildError LeanJaxpr) := do
  let jaxpr ←
    match LowerFnBody.run { declName := declName, params := params, body := body, hints := hints } with
    | .ok j => pure j
    | .error err => return .error (.conversion (toString err))

  match runValidation jaxpr with
  | .error err => return .error err
  | .ok () =>
    match (← runCoverage cfg jaxpr) with
    | .error err => return .error err
    | .ok () => return .ok jaxpr

/-- Convert + validate + optionally rule-check from GPU `KStmt` array. -/
def buildFromKStmts
    (cfg : BuildConfig := {})
    (stmts : Array Tyr.GPU.Codegen.KStmt) :
    CoreM (Except BuildError LeanJaxpr) := do
  let jaxpr ←
    match LowerKStmt.run { stmts := stmts } with
    | .ok j => pure j
    | .error err =>
      return .error (.conversion (toString err))

  match runValidation jaxpr with
  | .error err => return .error err
  | .ok () =>
    match (← runCoverage cfg jaxpr) with
    | .error err => return .error err
    | .ok () => return .ok jaxpr

private def extractAfterBuild
    (buildRes : Except BuildError LeanJaxpr) :
    CoreM (Except BuildExtractError (LeanJaxpr × Array LocalJacEdge)) := do
  match buildRes with
  | .error err =>
    return .error (.build err)
  | .ok jaxpr =>
    match (← extractLocalJacEdges jaxpr) with
    | .ok edges =>
      return .ok (jaxpr, edges)
    | .error errs =>
      return .error (.ruleExecution (errs.map ruleExecutionErrorToString))

/--
Convert + validate + optionally coverage-check + execute local-Jacobian rules
from a Lean IR declaration.
-/
def buildAndExtractFromDecl
    (cfg : BuildConfig := {})
    (decl : Decl) :
    CoreM (Except BuildExtractError (LeanJaxpr × Array LocalJacEdge)) := do
  extractAfterBuild (← buildFromDecl cfg decl)

/--
Convert + validate + optionally coverage-check + execute local-Jacobian rules
from raw `FnBody`.
-/
def buildAndExtractFromFnBody
    (cfg : BuildConfig := {})
    (declName : Name)
    (params : Array Param)
    (body : FnBody)
    (hints : FnBodyLoweringHints := {}) :
    CoreM (Except BuildExtractError (LeanJaxpr × Array LocalJacEdge)) := do
  extractAfterBuild (← buildFromFnBody cfg declName params body hints)

/--
Convert + validate + optionally coverage-check + execute local-Jacobian rules
from GPU `KStmt` array.
-/
def buildAndExtractFromKStmts
    (cfg : BuildConfig := {})
    (stmts : Array Tyr.GPU.Codegen.KStmt) :
    CoreM (Except BuildExtractError (LeanJaxpr × Array LocalJacEdge)) := do
  extractAfterBuild (← buildFromKStmts cfg stmts)

end Tyr.AD.JaxprLike
