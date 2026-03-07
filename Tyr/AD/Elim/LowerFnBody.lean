import Std.Data.HashSet
import Tyr.AD.Elim.LowerKStmt

/-!
# Tyr.AD.Elim.LowerFnBody

Conservative lowering from normalized `LeanJaxpr` into a Lean IR `FnBody`
plus parameter list, for the KStmt-op subset supported by `LowerKStmt`.
-/

namespace Tyr.AD.Elim

open Lean
open Lean.IR
open Tyr.AD.JaxprLike

/-- Pieces needed to feed `fromFnBody`/`buildFromFnBody` after lowering. -/
structure LoweredFnBody where
  declName : Name
  params : Array Param
  body : FnBody
  deriving Inhabited

def defaultLoweredFnDeclName : Name :=
  `Tyr.AD.Elim.loweredFnBody

private def mkVarId (v : JVar) : VarId :=
  { idx := v.id }

private def mkParam (v : JVar) : Param :=
  { x := mkVarId v, borrow := false, ty := v.ty }

private def loweringError (message : String) : String :=
  s!"lowerToFnBody: {message}"

private def eqnLoweringError (idx0 : Nat) (eqn : JEqn) (message : String) : String :=
  loweringError s!"eqn #{idx0} (`{eqn.op}`): {message}"

private def checkUniqueBinders
    (invars : Array JVar)
    (eqns : Array JEqn) :
    Array String := Id.run do
  let mut errors : Array String := #[]
  let mut seen : Std.HashSet Nat := {}

  for v in invars do
    if seen.contains v.id then
      errors := errors.push <|
        loweringError s!"duplicate parameter binder ID {v.id}"
    else
      seen := seen.insert v.id

  for h : idx0 in [:eqns.size] do
    let eqn := eqns[idx0]
    match eqn.outvars[0]? with
    | none =>
      errors := errors.push <|
        eqnLoweringError idx0 eqn "expected exactly one output variable, got none"
    | some outv =>
      if eqn.outvars.size != 1 then
        errors := errors.push <|
          eqnLoweringError idx0 eqn
            s!"expected exactly one output variable, got {eqn.outvars.size}"
      if seen.contains outv.id then
        errors := errors.push <|
          eqnLoweringError idx0 eqn s!"duplicate binder ID {outv.id}"
      else
        seen := seen.insert outv.id

  return errors

private def validateJaxprForFnBody (jaxpr : LeanJaxpr) : Array String := Id.run do
  let mut errors : Array String := #[]

  if !jaxpr.constvars.isEmpty then
    errors := errors.push <|
      loweringError s!"constvars are unsupported ({jaxpr.constvars.size} constvars found)"

  if jaxpr.outvars.size != 1 then
    errors := errors.push <|
      loweringError s!"expected exactly one terminal output variable, got {jaxpr.outvars.size}"

  for h : idx0 in [:jaxpr.eqns.size] do
    let eqn := jaxpr.eqns[idx0]
    if !isKStmtLowerableOpName eqn.op then
      errors := errors.push <|
        eqnLoweringError idx0 eqn "unsupported op name for FnBody lowering"
    if eqn.outvars.size != 1 then
      errors := errors.push <|
        eqnLoweringError idx0 eqn
          s!"expected exactly one output variable, got {eqn.outvars.size}"

  errors ++ checkUniqueBinders jaxpr.invars jaxpr.eqns

private def buildFnBody (jaxpr : LeanJaxpr) : FnBody := Id.run do
  let terminal : VarId := mkVarId jaxpr.outvars[0]!
  let mut body : FnBody := .ret (.var terminal)

  for revIdx in [:jaxpr.eqns.size] do
    let idx0 := jaxpr.eqns.size - 1 - revIdx
    let eqn := jaxpr.eqns[idx0]!
    let outv := eqn.outvars[0]!
    let lhs : VarId := mkVarId outv
    let args := eqn.invars.map (fun v => Arg.var (mkVarId v))
    body := .vdecl lhs outv.ty (.fap eqn.op args) body

  return body

/--
Lower a `LeanJaxpr` into Lean IR function-body pieces.
This emits only direct `.vdecl _ _ (.fap ..)` chains and a final `.ret`.
-/
def lowerToFnBody
    (jaxpr : LeanJaxpr)
    (declName : Name := defaultLoweredFnDeclName) :
    Except (Array String) LoweredFnBody := Id.run do
  let errors := validateJaxprForFnBody jaxpr
  if !errors.isEmpty then
    return .error errors

  return .ok {
    declName := declName
    params := jaxpr.invars.map mkParam
    body := buildFnBody jaxpr
  }

end Tyr.AD.Elim
