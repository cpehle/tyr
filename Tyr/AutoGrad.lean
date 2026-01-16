import Lean.Compiler.IR.Basic
import Lean.Compiler.IR.Format
import Lean.Compiler.IR.CompilerM
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Var
import Tyr.TensorStruct

namespace Tyr.AD

open torch (Static TensorStruct T)

open Lean.IR

/-! ## Parameter Kinds for Differentiation

Parameters can be:
- `diff`: Differentiable (receives tangent/cotangent)
- `static`: Non-differentiable, passed through unchanged (like `torch.Static`)
- `frozen`: Forward-only tensor, no gradient update (like `torch.Frozen`)
-/

/-- Describes how a parameter participates in differentiation -/
inductive ParamKind where
  | diff      -- Differentiable: needs tangent (JVP) or cotangent (VJP)
  | static    -- Non-differentiable: passed through unchanged
  | frozen    -- Forward-only: participates in forward, but no gradient update
  deriving Repr, BEq, Inhabited

/-! ## DifferentiableManifold Typeclass

A mathematically rigorous design based on differential geometry:

1. **The space (manifold M)** - The type α itself
2. **Tangent space (TₓM)** - For JVP/forward-mode, infinitesimal directions at a point
3. **Cotangent space (T*ₓM)** - For VJP/reverse-mode, linear functionals on tangent vectors
4. **Musical isomorphism (♭/♯)** - Mapping between tangent and cotangent via a metric
5. **Exponential map (exp)** - Moving a point in the space using a tangent vector

This enables proper Riemannian optimization on curved manifolds.
-/

/-- A differentiable manifold with tangent and cotangent bundles.

    Mathematically:
    - M is the manifold (type α)
    - TₓM is the tangent space at point x (Tangent x)
    - T*ₓM is the cotangent space at point x (Cotangent x)
    - ♯ (sharp): T*M → TM raises indices using the metric
    - ♭ (flat): TM → T*M lowers indices using the metric
    - exp: M × TM → M is the exponential map (or retraction)
-/
class DifferentiableManifold (M : Type) where
  /-- Tangent space at a point. For Euclidean spaces, this is M itself. -/
  Tangent : M → Type
  /-- Cotangent space at a point. For Euclidean spaces, this is M itself. -/
  Cotangent : M → Type

  /-- Zero tangent vector at a point -/
  zeroTangent (x : M) : Tangent x
  /-- Zero cotangent at a point -/
  zeroCotangent (x : M) : Cotangent x

  /-- Add tangent vectors (TₓM is a vector space) -/
  addTangent {x : M} : Tangent x → Tangent x → Tangent x
  /-- Add cotangents (T*ₓM is a vector space) -/
  addCotangent {x : M} : Cotangent x → Cotangent x → Cotangent x

  /-- Scale a tangent vector -/
  scaleTangent {x : M} (s : Float) : Tangent x → Tangent x

  /-- Musical isomorphism ♯ (sharp): Cotangent → Tangent
      Uses the Riemannian metric to "raise indices" -/
  sharp {x : M} : Cotangent x → Tangent x

  /-- Musical isomorphism ♭ (flat): Tangent → Cotangent
      Uses the Riemannian metric to "lower indices" -/
  flat {x : M} : Tangent x → Cotangent x

  /-- Exponential map: Move from x in direction v
      For Euclidean spaces: exp(x, v) = x + v
      For curved manifolds: follows geodesic from x with initial velocity v -/
  exp (x : M) : Tangent x → M

  /-- Retraction (computationally cheaper approximation to exp) -/
  retract (x : M) : Tangent x → M := exp x

namespace DifferentiableManifold

/-- Simplified typeclass for Euclidean spaces where T = T* = M.
    These spaces have a flat metric where sharp/flat are identity. -/
class EuclideanSpace (M : Type) where
  /-- Zero element -/
  zero : M
  /-- Addition -/
  add : M → M → M
  /-- Scalar multiplication -/
  smul : Float → M → M
  /-- Inner product (defines the Euclidean metric) -/
  inner : M → M → Float

/-- Euclidean spaces are automatically DifferentiableManifolds -/
instance euclideanManifold [E : EuclideanSpace M] : DifferentiableManifold M where
  Tangent _ := M
  Cotangent _ := M
  zeroTangent _ := E.zero
  zeroCotangent _ := E.zero
  addTangent := E.add
  addCotangent := E.add
  scaleTangent s v := E.smul s v
  -- For Euclidean metric, sharp and flat are identity
  sharp := id
  flat := id
  -- exp(x, v) = x + v
  exp x v := E.add x v

/-- Float is a Euclidean space (1-dimensional) -/
instance floatEuclidean : EuclideanSpace Float where
  zero := 0.0
  add := (· + ·)
  smul s x := s * x
  inner a b := a * b

/-- Static types are 0-dimensional manifolds (no tangent directions) -/
instance staticManifold {α : Type} : DifferentiableManifold (Static α) where
  Tangent _ := Unit
  Cotangent _ := Unit
  zeroTangent _ := ()
  zeroCotangent _ := ()
  addTangent _ _ := ()
  addCotangent _ _ := ()
  scaleTangent _ _ := ()
  sharp := id
  flat := id
  exp x _ := x  -- Static values don't move

/-- TensorStruct types form Euclidean spaces (product manifold).
    tangent = cotangent = primal type, using TensorStruct.zipWith for operations. -/
instance tensorStructEuclidean [s : TensorStruct α] [Inhabited α] : EuclideanSpace α where
  zero := default
  add := TensorStruct.zipWith torch.add
  smul scalar m := TensorStruct.map (torch.mul_scalar · scalar) m
  -- Inner product: sum of all element-wise products across tensors
  inner a b :=
    let product := TensorStruct.zipWith torch.mul a b
    TensorStruct.fold (fun t acc => acc + torch.nn.item (torch.nn.sumAll t)) 0.0 product

/-- Gradient descent step using exponential map / retraction.
    For Euclidean spaces: x' = x - lr * grad
    For curved manifolds: x' = retract(x, -lr * sharp(grad)) -/
def gradientStep [DifferentiableManifold M]
    (x : M) (grad : Cotangent x) (lr : Float) : M :=
  let tangent := sharp grad
  let negTangent := scaleTangent (-lr) tangent
  retract x negTangent

end DifferentiableManifold

-- Backward compatibility aliases
abbrev Differentiable := DifferentiableManifold

/-- Replace variable in Argument -/
def replaceArg (a : Arg) (old new : VarId) : Arg :=
  match a with
  | .var x => if x.idx == old.idx then .var new else a
  | .erased => .erased

/-- Replace variable in Expression -/
def replaceExpr (e : Expr) (old new : VarId) : Expr :=
  match e with
  | .ctor i args => .ctor i (args.map (replaceArg · old new))
  | .reset n x => .reset n (if x.idx == old.idx then new else x)
  | .reuse x i u args => .reuse (if x.idx == old.idx then new else x) i u (args.map (replaceArg · old new))
  | .proj i x => .proj i (if x.idx == old.idx then new else x)
  | .uproj i x => .uproj i (if x.idx == old.idx then new else x)
  | .sproj n o x => .sproj n o (if x.idx == old.idx then new else x)
  | .fap c args => .fap c (args.map (replaceArg · old new))
  | .pap c args => .pap c (args.map (replaceArg · old new))
  | .ap x args => .ap (if x.idx == old.idx then new else x) (args.map (replaceArg · old new))
  | .box t x => .box t (if x.idx == old.idx then new else x)
  | .unbox x => .unbox (if x.idx == old.idx then new else x)
  | .lit l => .lit l
  | .isShared x => .isShared (if x.idx == old.idx then new else x)

/-- Replace variable in FnBody -/
partial def replaceVar (b : FnBody) (old new : VarId) : FnBody :=
  match b with
  | .vdecl x ty e rest => 
    .vdecl x ty (replaceExpr e old new) (replaceVar rest old new)
  | .jdecl j xs v rest => 
    .jdecl j xs (replaceVar v old new) (replaceVar rest old new)
  | .set x i y rest => 
    .set (if x.idx == old.idx then new else x) i (replaceArg y old new) (replaceVar rest old new)
  | .setTag x i rest => 
    .setTag (if x.idx == old.idx then new else x) i (replaceVar rest old new)
  | .uset x i y rest => 
    .uset (if x.idx == old.idx then new else x) i (if y.idx == old.idx then new else y) (replaceVar rest old new)
  | .sset x i o y ty rest => 
    .sset (if x.idx == old.idx then new else x) i o (if y.idx == old.idx then new else y) ty (replaceVar rest old new)
  | .inc x n c p rest => 
    .inc (if x.idx == old.idx then new else x) n c p (replaceVar rest old new)
  | .dec x n c p rest => 
    .dec (if x.idx == old.idx then new else x) n c p (replaceVar rest old new)
  | .del x rest => 
    .del (if x.idx == old.idx then new else x) (replaceVar rest old new)
  | .case tid x xType alts => 
    .case tid (if x.idx == old.idx then new else x) xType (alts.map fun 
      | .ctor info b => .ctor info (replaceVar b old new)
      | .default b => .default (replaceVar b old new))
  | .jmp j args => 
    .jmp j (args.map (replaceArg · old new))
  | .ret x => 
    .ret (replaceArg x old new)
  | .unreachable => .unreachable

-- AD Context and Monad
structure ADContext where
  tangents : Std.HashMap Nat Nat := {}
  cotangents : Std.HashMap Nat Nat := {}
  nextVarIdx : Nat := 0
  /-- Accumulator for backward pass -/
  pullbackStack : Array (FnBody → FnBody) := #[]
  /-- Variables that are static (non-differentiable) - skip tangent/cotangent -/
  staticVars : Std.HashSet Nat := {}
  deriving Inhabited

abbrev ADM := StateT ADContext Lean.CoreM

def getFreshVarIdx : ADM Nat := do
  let s ← get
  let idx := s.nextVarIdx
  modify fun s => { s with nextVarIdx := idx + 1 }
  return idx

def getFreshVar : ADM VarId := do
  return { idx := ← getFreshVarIdx }

def getFreshGpuVar : ADM Tyr.GPU.Codegen.VarId := do
  return { idx := ← getFreshVarIdx }

def getTangentIdx (idx : Nat) : ADM Nat := do
  let s ← get
  match s.tangents.get? idx with
  | some t => return t
  | none => return idx 

def getTangent (x : VarId) : ADM VarId := do
  return { idx := ← getTangentIdx x.idx }

def getGpuTangent (x : Tyr.GPU.Codegen.VarId) : ADM Tyr.GPU.Codegen.VarId := do
  return { idx := ← getTangentIdx x.idx }

def setTangentIdx (x idx : Nat) : ADM Unit := do
  modify fun s => { s with tangents := s.tangents.insert x idx }

def setTangent (x dx : VarId) : ADM Unit := do
  setTangentIdx x.idx dx.idx

def setGpuTangent (x dx : Tyr.GPU.Codegen.VarId) : ADM Unit := do
  setTangentIdx x.idx dx.idx

def getCotangentIdx (idx : Nat) : ADM Nat := do
  let s ← get
  match s.cotangents.get? idx with
  | some dx => return dx
  | none => do
    let dx ← getFreshVarIdx
    modify fun s => { s with cotangents := s.cotangents.insert idx dx }
    return dx

def getCotangent (x : VarId) : ADM VarId := do
  return { idx := ← getCotangentIdx x.idx }

def getGpuCotangent (x : Tyr.GPU.Codegen.VarId) : ADM Tyr.GPU.Codegen.VarId := do
  return { idx := ← getCotangentIdx x.idx }

def setCotangentIdx (x idx : Nat) : ADM Unit := do
  modify fun s => { s with cotangents := s.cotangents.insert x idx }

def setCotangent (x dx : VarId) : ADM Unit := do
  setCotangentIdx x.idx dx.idx

def setGpuCotangent (x dx : Tyr.GPU.Codegen.VarId) : ADM Unit := do
  setCotangentIdx x.idx dx.idx

/-- Check if a variable is marked as static (non-differentiable) -/
def isStaticVar (idx : Nat) : ADM Bool := do
  return (← get).staticVars.contains idx

/-- Mark a variable as static (non-differentiable) -/
def markStatic (idx : Nat) : ADM Unit := do
  modify fun s => { s with staticVars := s.staticVars.insert idx }

/-- Mark a VarId as static -/
def markStaticVar (x : VarId) : ADM Unit := markStatic x.idx

-- AD Rules for Lean IR
abbrev JVPBuilder := FnBody → FnBody
abbrev JVPRule := Array Arg → Array Arg → IRType → ADM (JVPBuilder × VarId × VarId)

abbrev VJPBuilder := FnBody → FnBody
abbrev VJPRule := Array Arg → VarId → IRType → ADM (VJPBuilder × Array VarId)

structure ADRegistry where
  jvpRules : Std.HashMap Lean.Name JVPRule := {}
  vjpRules : Std.HashMap Lean.Name VJPRule := {}
  deriving Inhabited

initialize adRegistry : Lean.EnvExtension ADRegistry ←
  Lean.registerEnvExtension (pure {})

-- GPU AD Registry
/-- 
  GPU VJP Rule Type:
  Arguments:
  - dout : Cotangent of the output
  - dins : Array of cotangents of the inputs
  - ctx  : Array of primal variables preserved from forward pass
  Returns:
  - Array KStmt: The statements implementing the backward pass
-/
abbrev GpuVJPRule := Tyr.GPU.Codegen.VarId → Array Tyr.GPU.Codegen.VarId → Array Tyr.GPU.Codegen.VarId → ADM (Array Tyr.GPU.Codegen.KStmt)

/--
  GPU AD Registry for custom operations.
  Unlike Lean IR which uses Name, GPU IR operations are typically Enums (BinaryOp, UnaryOp).
  We map a String key (e.g. "UnaryOp.Sin") to the rule.
-/
structure GpuADRegistry where
  /-- Map from Op Name to VJP Rule -/
  vjpRules : Std.HashMap String GpuVJPRule := {}
  deriving Inhabited

initialize gpuAdRegistry : Lean.EnvExtension GpuADRegistry ←
  Lean.registerEnvExtension (pure {})

def registerJVPRule (fn : Lean.Name) (rule : JVPRule) : Lean.CoreM Unit := do
  Lean.modifyEnv fun env => adRegistry.modifyState env fun s => 
    { s with jvpRules := s.jvpRules.insert fn rule }

def registerVJPRule (fn : Lean.Name) (rule : VJPRule) : Lean.CoreM Unit := do
  Lean.modifyEnv fun env => adRegistry.modifyState env fun s => 
    { s with vjpRules := s.vjpRules.insert fn rule }

def getJVPRule (fn : Lean.Name) : Lean.CoreM (Option JVPRule) := do
  return (adRegistry.getState (← Lean.getEnv)).jvpRules.get? fn

def getVJPRule (fn : Lean.Name) : Lean.CoreM (Option VJPRule) := do
  return (adRegistry.getState (← Lean.getEnv)).vjpRules.get? fn

-- Lean VJP Registration Helper

partial def unpackTuple (res : VarId) (vars : List VarId) : ADM (FnBody → FnBody) := do
  match vars with
  | [] => return id
  | [v] => 
    return fun b => FnBody.vdecl v IRType.object (Expr.fap `id #[Arg.var res]) b

  | v :: vs =>
    let tail ← getFreshVar
    let unpackTail ← unpackTuple tail vs
    return fun b => 
      FnBody.vdecl v IRType.object (Expr.proj 0 res) <|
      FnBody.vdecl tail IRType.object (Expr.proj 1 res) <|
      unpackTail b

/-- Register a VJP rule with parameter kinds.
    Static parameters are excluded from cotangent computation. -/
def registerLeanVJPRuleWithKinds (primalFn vjpFn : Lean.Name)
    (paramKinds : Array ParamKind := #[]) : Lean.CoreM Unit := do
  registerVJPRule primalFn fun args dy _resTy => do
    let res ← getFreshVar
    let callArgs := args.push (Arg.var dy)

    let mut dxs := #[]
    let mut varsToUnpack := []

    -- Create fresh vars for cotangents (only for diff params)
    -- For static params, create a dummy var that will be skipped
    for i in [:args.size] do
      let kind := paramKinds.getD i .diff
      if kind == .diff then
        let dx ← getFreshVar
        dxs := dxs.push dx
        varsToUnpack := varsToUnpack ++ [dx]
      else
        -- Static/frozen params: create dummy var and mark as static
        let dummyVar ← getFreshVar
        markStatic dummyVar.idx
        dxs := dxs.push dummyVar
        -- Also mark the arg var itself as static if it's a var
        match args[i]! with
        | .var v => markStatic v.idx
        | _ => pure ()

    let unpacker ← unpackTuple res varsToUnpack

    let builder := fun rest =>
      FnBody.vdecl res IRType.object (Expr.fap vjpFn callArgs) <|
      unpacker rest

    return (builder, dxs)

/-- Register a VJP rule (all params differentiable) -/
def registerLeanVJPRule (primalFn vjpFn : Lean.Name) : Lean.CoreM Unit := do
  registerLeanVJPRuleWithKinds primalFn vjpFn #[]

/-- Register a JVP rule with parameter kinds.
    Static parameters don't receive tangent arguments. -/
def registerLeanJVPRuleWithKinds (primalFn jvpFn : Lean.Name)
    (paramKinds : Array ParamKind := #[]) : Lean.CoreM Unit := do
  registerJVPRule primalFn fun args tans _resTy => do
    let resPair ← getFreshVar

    -- Build call args: primal args interleaved with tangents for diff params only
    let mut callArgs : Array Arg := #[]
    let mut tanIdx := 0
    for i in [:args.size] do
      callArgs := callArgs.push args[i]!
      let kind := paramKinds.getD i .diff
      if kind == .diff then
        -- Only add tangent for differentiable params
        if h : tanIdx < tans.size then
          callArgs := callArgs.push tans[tanIdx]
          tanIdx := tanIdx + 1
      -- static/frozen params: no tangent arg added

    let prim ← getFreshVar
    let tan ← getFreshVar

    let builder := fun rest =>
      FnBody.vdecl resPair IRType.object (Expr.fap jvpFn callArgs) <|
      FnBody.vdecl prim IRType.object (Expr.proj 0 resPair) <|
      FnBody.vdecl tan IRType.object (Expr.proj 1 resPair) <|
      rest

    return (builder, prim, tan)

/-- Register a JVP rule (all params differentiable) -/
def registerLeanJVPRule (primalFn jvpFn : Lean.Name) : Lean.CoreM Unit := do
  registerLeanJVPRuleWithKinds primalFn jvpFn #[]

-- JVP Interpreter
partial def jvp (b : FnBody) : ADM FnBody := do
  match b with
  | .vdecl x ty e rest =>
    match e with
    | .fap fn args =>
      let rule? ← getJVPRule fn
      match rule? with
      | some rule =>
        -- Only get tangents for non-static args
        let tanArgs ← args.filterMapM fun
          | Arg.var v => do
            if (← isStaticVar v.idx) then
              return none  -- Skip tangent for static vars
            else
              return some (Arg.var (← getTangent v))
          | Arg.erased => return none
        let (builder, prim, tan) ← rule args tanArgs ty
        setTangent x tan
        let rest' ← jvp rest
        return builder (replaceVar rest' x prim)
      | none =>
        let rest' ← jvp rest
        return .vdecl x ty e rest'
    | _ =>
      let rest' ← jvp rest
      return .vdecl x ty e rest'
  | .ret (.var x) =>
    let dx ← getTangent x
    let pair ← getFreshVar
    let ctorInfo : CtorInfo := { name := `Prod.mk, cidx := 0, size := 2, usize := 0, ssize := 0 }
    return .vdecl pair IRType.object (.ctor ctorInfo #[Arg.var x, .var dx]) (.ret (.var pair))
  | other => return other

-- VJP Interpreter
partial def vjp (b : FnBody) : ADM FnBody := do
  match b with
  | .vdecl x ty e rest =>
    match e with
    | .fap fn args =>
      let rule? ← getVJPRule fn
      match rule? with
      | some rule =>
        let dx ← getCotangent x
        let (pbBuilder, argCotans) ← rule args dx ty
        
        -- Register cotangents for arguments (skip static vars)
        for i in [:args.size] do
           match args[i]! with
           | .var v =>
             -- Skip cotangent for static vars
             if !(← isStaticVar v.idx) then
               setCotangent v argCotans[i]!
           | _ => pure ()

        modify fun s => { s with pullbackStack := s.pullbackStack.push pbBuilder }
        let rest' ← vjp rest
        return .vdecl x ty e rest'
      | none =>
        let rest' ← vjp rest
        return .vdecl x ty e rest'
    | _ =>
      let rest' ← vjp rest
      return .vdecl x ty e rest'
  | .ret (.var x) =>
    let dx ← getCotangent x
    let s ← get
    -- Construct backward pass
    let _backward := s.pullbackStack.foldr (fun pb acc => pb acc) (.ret (.var dx))
    
    return .ret (.var x)
  | other => return other

-- Linearize
def linearize (decl : Decl) : ADM Decl := do
  match decl with
  | Decl.fdecl f params _ty body info =>
    let tanParams ← params.mapM fun p => do
      let dp ← getFreshVar
      setTangent p.x dp
      return { p with x := dp } 
    let body' ← jvp body
    return Decl.fdecl (Lean.Name.mkStr f "jvp") (params ++ tanParams) IRType.object body' info
  | other => return other

-- GPU Registration helpers
def registerGpuVJPRule (opName : String) (rule : GpuVJPRule) : Lean.CoreM Unit := do
  Lean.modifyEnv fun env => gpuAdRegistry.modifyState env fun s => 
    { s with vjpRules := s.vjpRules.insert opName rule }

def getGpuVJPRule (opName : String) : Lean.CoreM (Option GpuVJPRule) := do
  return (gpuAdRegistry.getState (← Lean.getEnv)).vjpRules.get? opName

-- Attributes

/-- Parse a list of static parameter indices into ParamKind array -/
def parseStaticIndices (staticIndices : Array Nat) (numParams : Nat := 10) : Array ParamKind :=
  Array.range numParams |>.map fun i =>
    if staticIndices.contains i then .static else .diff

-- Basic syntax: @[jvp primalFn] or @[vjp primalFn]
syntax (name := jvpAttr) "jvp" ident : attr
syntax (name := vjpAttr) "vjp" ident : attr

-- Extended syntax with static params: @[jvp primalFn, static := [1, 3]]
syntax (name := jvpAttrStatic) "jvp" ident "," "static" ":=" "[" num,* "]" : attr
syntax (name := vjpAttrStatic) "vjp" ident "," "static" ":=" "[" num,* "]" : attr

initialize
  -- Basic JVP (all params differentiable)
  Lean.registerBuiltinAttribute {
    name := `jvpAttr
    descr := "Register JVP rule"
    add := fun declName stx _ => do
      match stx with
      | `(attr| jvp $id) => registerLeanJVPRule id.getId declName
      | _ => throwError "invalid jvp attribute"
  }

  -- Basic VJP (all params differentiable)
  Lean.registerBuiltinAttribute {
    name := `vjpAttr
    descr := "Register VJP rule"
    add := fun declName stx _ => do
      match stx with
      | `(attr| vjp $id) => registerLeanVJPRule id.getId declName
      | _ => throwError "invalid vjp attribute"
  }

  -- JVP with static params
  Lean.registerBuiltinAttribute {
    name := `jvpAttrStatic
    descr := "Register JVP rule with static (non-differentiable) parameters"
    add := fun declName stx _ => do
      match stx with
      | `(attr| jvp $id, static := [$indices,*]) =>
        let staticIndices := indices.getElems.map fun n => n.getNat
        let paramKinds := parseStaticIndices staticIndices
        registerLeanJVPRuleWithKinds id.getId declName paramKinds
      | _ => throwError "invalid jvp attribute with static"
  }

  -- VJP with static params
  Lean.registerBuiltinAttribute {
    name := `vjpAttrStatic
    descr := "Register VJP rule with static (non-differentiable) parameters"
    add := fun declName stx _ => do
      match stx with
      | `(attr| vjp $id, static := [$indices,*]) =>
        let staticIndices := indices.getElems.map fun n => n.getNat
        let paramKinds := parseStaticIndices staticIndices
        registerLeanVJPRuleWithKinds id.getId declName paramKinds
      | _ => throwError "invalid vjp attribute with static"
  }

end Tyr.AD