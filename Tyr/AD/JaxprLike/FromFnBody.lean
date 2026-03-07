import Tyr.AD.JaxprLike.Core
import Tyr.AD.JaxprLike.KStmtNames

/-!
# Tyr.AD.JaxprLike.FromFnBody

Conservative conversion skeleton from `Lean.IR.FnBody` to `LeanJaxpr`.
Only direct `Expr.fap` value declarations are lowered into equations.
All unsupported constructs are collected as diagnostics and reported as an `Except` error.
-/

namespace Tyr.AD.JaxprLike

open Lean
open Lean.IR

/-- Conversion diagnostic with lightweight source information. -/
structure FromFnBodyDiagnostic where
  source : SourceRef := {}
  message : String
  deriving Repr, Inhabited

def formatSourceRef (source : SourceRef) : String :=
  let declName :=
    if source.decl == .anonymous then "<unknown>" else toString source.decl
  let linePart :=
    match source.line? with
    | some line => s!":{line}"
    | none => ""
  let colPart :=
    match source.col? with
    | some col => s!":{col}"
    | none => ""
  s!"{declName}{linePart}{colPart}"

def FromFnBodyDiagnostic.toString (diag : FromFnBodyDiagnostic) : String :=
  s!"{formatSourceRef diag.source}: {diag.message}"

instance : ToString FromFnBodyDiagnostic := ⟨FromFnBodyDiagnostic.toString⟩

/-- Conversion context; source mapping is intentionally minimal at this stage. -/
structure FromFnBodyCtx where
  source : SourceRef := {}
  deriving Inhabited

/-- Stateful accumulator used while traversing a `FnBody`. -/
structure FromFnBodyState where
  varEnv : Std.HashMap Nat JVar := {}
  nextId : Nat := 1
  eqns : Array JEqn := #[]
  outvars : Array JVar := #[]
  diagnostics : Array FromFnBodyDiagnostic := #[]
  deriving Inhabited

abbrev FromFnBodyM := StateM FromFnBodyState

def addDiagnostic (ctx : FromFnBodyCtx) (message : String) : FromFnBodyM Unit := do
  modify fun st =>
    { st with diagnostics := st.diagnostics.push { source := ctx.source, message := message } }

/--
Bind an IR `VarId` into the lowering environment using fresh, deterministic
`JVarId`s. This mirrors Lean IR-construction style where binders are assigned
fresh indices by the lowering pass instead of reusing source IDs directly.
-/
def bindVar (ctx : FromFnBodyCtx) (x : VarId) (ty : IRType) : FromFnBodyM JVar := do
  let st ← get
  match st.varEnv.get? x.idx with
  | some existing =>
    addDiagnostic ctx s!"Duplicate binder x_{x.idx} encountered during `FnBody` lowering."
    pure existing
  | none =>
    let v : JVar := { id := st.nextId, ty := ty }
    set { st with varEnv := st.varEnv.insert x.idx v, nextId := st.nextId + 1 }
    pure v

private def expectedArity? (op : OpName) : Option Nat :=
  if isStopGradientOpName op then some 1
  else if isIotaOpName op then some 0
  else if isDevicePutOpName op then some 1
  else if isCommunicationUnaryOpName op then some 1
  else if isPadAliasOpName op then some 2
  else if isSelectFixedArityAliasOpName op then some 3
  else if isDynamicUpdateIndexInDimAliasOpName op then some 3
  else if isStructuralUnaryAliasOpName op then some 1
  else if isReductionUnaryAliasOpName op then some 1
  else if isDotGeneralOpName op then some 2
  else none

private def expectedMinArity? (op : OpName) : Option Nat :=
  if isCondAliasOpName op then some 2
  else if isScanAliasOpName op then some 1
  else none

private def loweringKindForOp (op : OpName) : Name :=
  if isStopGradientOpName op then
    `fnbody.vdecl.fap.stop_gradient
  else if isIotaOpName op then
    `fnbody.vdecl.fap.iota
  else if isDevicePutOpName op then
    `fnbody.vdecl.fap.device_put
  else if isPjitOpName op then
    `fnbody.vdecl.fap.pjit
  else if isCommunicationUnaryOpName op then
    `fnbody.vdecl.fap.communication
  else if isHigherOrderControlAliasOpName op then
    `fnbody.vdecl.fap.control_flow
  else if isStructuralUnaryAliasOpName op || isPadAliasOpName op ||
      isConcatLikeAliasOpName op ||
      isReductionUnaryAliasOpName op || isSelectNAliasOpName op ||
      isDynamicProjectionAliasOpName op || isDynamicUpdateAliasOpName op then
    `fnbody.vdecl.fap.structural
  else if isDotGeneralOpName op then
    `fnbody.vdecl.fap.dot_general
  else
    `fnbody.vdecl.fap

private def canonicalOpName (op : OpName) : OpName :=
  if isDotGeneralOpName op then
    kstmtDotGeneralOpName
  else if isScanAliasOpName op then
    scanAliasOpName
  else if isCondAliasOpName op then
    condAliasOpName
  else
    op

/-- Extract a lowered variable from an IR argument when possible. -/
def argToJVar? (arg : Arg) : FromFnBodyM (Except String JVar) := do
  match arg with
  | .var x =>
    match (← get).varEnv.get? x.idx with
    | some v => pure (.ok v)
    | none => pure (.error s!"unbound variable x_{x.idx}")
  | .erased =>
    pure (.error "erased argument")

/-- Convert argument arrays to lowered variables, recording diagnostics on failure. -/
def argsToJVars? (ctx : FromFnBodyCtx) (args : Array Arg) : FromFnBodyM (Option (Array JVar)) := do
  let mut ok := true
  let mut invars : Array JVar := #[]
  for i in [:args.size] do
    match (← argToJVar? args[i]!) with
    | .ok v => invars := invars.push v
    | .error err =>
      ok := false
      addDiagnostic ctx s!"Unsupported function-application argument at index {i}: {err}."
  if ok then
    pure (some invars)
  else
    pure none

private def controlArgsToJVars?
    (ctx : FromFnBodyCtx)
    (args : Array Arg) :
    FromFnBodyM (Option (Array JVar × Nat)) := do
  let mut invars : Array JVar := #[]
  let mut staticCount : Nat := 0
  for i in [:args.size] do
    match args[i]! with
    | .var x =>
      match (← get).varEnv.get? x.idx with
      | some v => invars := invars.push v
      | none =>
        -- Treat unbound vars in higher-order/control calls as static handles.
        staticCount := staticCount + 1
    | .erased =>
      -- Erased args are static handles (e.g., branch/body closures).
      staticCount := staticCount + 1
  pure (some (invars, staticCount))

private def argsToJVarsForOp?
    (ctx : FromFnBodyCtx)
    (op : OpName)
    (args : Array Arg) :
    FromFnBodyM (Option (Array JVar × Nat)) := do
  if isHigherOrderControlAliasOpName op then
    controlArgsToJVars? ctx args
  else
    match (← argsToJVars? ctx args) with
    | some invars => pure (some (invars, 0))
    | none => pure none

private def dotGeneralDefaultParams (rawOp canonicalOp : OpName) : OpParams :=
  if canonicalOp == kstmtDotGeneralOpName then
    #[
      OpParam.mkName .variant (if rawOp == canonicalOp then `generic else rawOp),
      OpParam.mkNats .lhsContract #[],
      OpParam.mkNats .rhsContract #[],
      OpParam.mkNats .lhsBatch #[],
      OpParam.mkNats .rhsBatch #[]
    ]
  else
    #[]

private def controlFlowParams
    (canonicalOp : OpName)
    (invars : Array JVar)
    (staticCount : Nat) : OpParams :=
  if isCondAliasOpName canonicalOp then
    let predCount := if invars.isEmpty then 0 else 1
    let dataCount := invars.size - predCount
    #[
      OpParam.mkNat .controlStaticArgCount staticCount,
      OpParam.mkNat .condPredicateCount predCount,
      OpParam.mkNat .condDataInputCount dataCount
    ]
  else if isScanAliasOpName canonicalOp then
    let carryCount := if invars.isEmpty then 0 else 1
    let dataCount := invars.size - carryCount
    #[
      OpParam.mkNat .controlStaticArgCount staticCount,
      OpParam.mkNat .scanCarryInputCount carryCount,
      OpParam.mkNat .scanDataInputCount dataCount,
      OpParam.mkNat .scanCarryOutputCount carryCount
    ]
  else
    #[]

def exprKind : IR.Expr → String
  | .ctor _ _ => "Expr.ctor"
  | .reset _ _ => "Expr.reset"
  | .reuse _ _ _ _ => "Expr.reuse"
  | .proj _ _ => "Expr.proj"
  | .uproj _ _ => "Expr.uproj"
  | .sproj _ _ _ => "Expr.sproj"
  | .fap _ _ => "Expr.fap"
  | .pap _ _ => "Expr.pap"
  | .ap _ _ => "Expr.ap"
  | .box _ _ => "Expr.box"
  | .unbox _ => "Expr.unbox"
  | .lit _ => "Expr.lit"
  | .isShared _ => "Expr.isShared"

/-- Seed the conversion environment from function parameters. -/
def registerInvars (ctx : FromFnBodyCtx) (params : Array Param) : FromFnBodyM (Array JVar) := do
  let mut invars : Array JVar := #[]
  for p in params do
    let v ← bindVar ctx p.x p.ty
    invars := invars.push v
  pure invars

/--
Traverse `FnBody` and lower supported nodes.
Only `.vdecl _ _ (.fap ..)` and terminal `.ret` are lowered.
-/
def traverseFnBody (ctx : FromFnBodyCtx) (body : FnBody) : FromFnBodyM Unit := do
  match body with
  | .vdecl x ty e rest =>
    let outvar ← bindVar ctx x ty
    match e with
    | .fap op args =>
      let canonicalOp := canonicalOpName op
      match expectedArity? op with
      | some nExpected =>
        if args.size != nExpected then
          addDiagnostic ctx
            s!"Primitive `{op}` expects arity {nExpected}, got {args.size}."
        else
          pure ()
      | none => pure ()
      match expectedMinArity? op with
      | some nMin =>
        if args.size < nMin then
          addDiagnostic ctx
            s!"Primitive `{op}` expects at least arity {nMin}, got {args.size}."
        else
          pure ()
      | none => pure ()
      let invars? ← argsToJVarsForOp? ctx canonicalOp args
      match invars? with
      | some (invars, staticCount) =>
        let extraParams :=
          controlFlowParams canonicalOp invars staticCount ++
            dotGeneralDefaultParams op canonicalOp
        modify fun st =>
          { st with
              eqns := st.eqns.push {
                op := canonicalOp
                invars := invars
                outvars := #[outvar]
                params := #[
                  OpParam.mkName .loweringKind (loweringKindForOp op),
                  OpParam.mkNat .fnbodyOutVarIdx outvar.id
                ] ++ extraParams
                source := ctx.source
              }
          }
      | none => pure ()
    | _ =>
      addDiagnostic ctx
        s!"Unsupported expression `{exprKind e}` in `FnBody.vdecl` for x_{x.idx}; only `Expr.fap` is lowered."
    traverseFnBody ctx rest

  | .jdecl j _ _ rest =>
    addDiagnostic ctx s!"Unsupported construct `FnBody.jdecl` (join point block_{j.idx})."
    traverseFnBody ctx rest

  | .set _ _ _ rest =>
    addDiagnostic ctx "Unsupported construct `FnBody.set` (effectful update)."
    traverseFnBody ctx rest

  | .setTag _ _ rest =>
    addDiagnostic ctx "Unsupported construct `FnBody.setTag` (effectful update)."
    traverseFnBody ctx rest

  | .uset _ _ _ rest =>
    addDiagnostic ctx "Unsupported construct `FnBody.uset` (effectful update)."
    traverseFnBody ctx rest

  | .sset _ _ _ _ _ rest =>
    addDiagnostic ctx "Unsupported construct `FnBody.sset` (effectful update)."
    traverseFnBody ctx rest

  | .inc _ _ _ _ rest =>
    addDiagnostic ctx "Unsupported construct `FnBody.inc` (reference-count effect)."
    traverseFnBody ctx rest

  | .dec _ _ _ _ rest =>
    addDiagnostic ctx "Unsupported construct `FnBody.dec` (reference-count effect)."
    traverseFnBody ctx rest

  | .del _ rest =>
    addDiagnostic ctx "Unsupported construct `FnBody.del` (lifetime effect)."
    traverseFnBody ctx rest

  | .case _ _ _ _ =>
    addDiagnostic ctx "Unsupported construct `FnBody.case` (control flow)."

  | .jmp j _ =>
    addDiagnostic ctx s!"Unsupported construct `FnBody.jmp` to block_{j.idx}."

  | .ret arg =>
    match (← argToJVar? arg) with
    | .ok outvar =>
      -- Keep the terminal return values exactly as observed at the final `ret`.
      modify fun st => { st with outvars := #[outvar] }
    | .error err =>
      addDiagnostic ctx s!"Unsupported return argument: {err}."

  | .unreachable =>
    addDiagnostic ctx "Unsupported construct `FnBody.unreachable`."

def renderDiagnostics (diagnostics : Array FromFnBodyDiagnostic) : String :=
  String.intercalate "\n" (diagnostics.toList.map (fun d => toString d))

/-- Convert a function body to `LeanJaxpr` using a conservative lowering strategy. -/
def fromFnBody
    (declName : Name)
    (params : Array Param)
    (body : FnBody) :
    Except String LeanJaxpr := Id.run do
  let ctx : FromFnBodyCtx := { source := { decl := declName } }
  let (invars, stAfterParams) := (registerInvars ctx params).run {}
  let (_, st) := (traverseFnBody ctx body).run stAfterParams

  if !st.diagnostics.isEmpty then
    return .error <|
      "FnBody -> LeanJaxpr conversion failed with unsupported constructs:\n" ++
      renderDiagnostics st.diagnostics

  if st.outvars.isEmpty then
    return .error s!"FnBody -> LeanJaxpr conversion failed: no terminal `ret` output found for `{declName}`."

  return .ok {
    constvars := #[]
    invars := invars
    eqns := st.eqns
    outvars := st.outvars
  }

/-- Helper to extract the traversable pieces from an IR declaration. -/
def declFnBodyView? (decl : Decl) : Except String (Name × Array Param × FnBody) :=
  match decl with
  | .fdecl f params _ body _ => .ok (f, params, body)
  | .extern f _ _ _ =>
    .error s!"FnBody -> LeanJaxpr conversion does not support extern declaration `{f}`."

/-- Convert an IR declaration to `LeanJaxpr` when it has an explicit body. -/
def fromDecl (decl : Decl) : Except String LeanJaxpr := do
  let (declName, params, body) ← declFnBodyView? decl
  fromFnBody declName params body

end Tyr.AD.JaxprLike
