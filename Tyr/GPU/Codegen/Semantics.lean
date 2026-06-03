import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Var

/-!
# Tyr.GPU.Codegen.Semantics

Structured scalar expressions and thread predicates used by GPU statements,
proof obligations, and proof-aware optimization passes.
-/

namespace Tyr.GPU.Codegen

/-- Fixed thread-block model used when discharging thread-participation facts. -/
structure ThreadCtx where
  blockDimX : Nat := 256
  blockDimY : Nat := 1
  blockDimZ : Nat := 1
  warpSize : Nat := 32
  warpGroupSize : Nat := 128
  deriving Repr, BEq

instance : Inhabited ThreadCtx where
  default := {}

namespace ThreadCtx

def totalThreads (ctx : ThreadCtx) : Nat :=
  ctx.blockDimX * ctx.blockDimY * ctx.blockDimZ

def axisBound (ctx : ThreadCtx) : Nat -> Nat
  | 0 => ctx.blockDimX
  | 1 => ctx.blockDimY
  | _ => ctx.blockDimZ

def linearThreadId (ctx : ThreadCtx) (tx ty tz : Nat) : Nat :=
  tx + ctx.blockDimX * (ty + ctx.blockDimY * tz)

def warpGroupOf (ctx : ThreadCtx) (tx ty tz : Nat) : Nat :=
  ctx.linearThreadId tx ty tz / ctx.warpGroupSize

end ThreadCtx

mutual

/-- Symbolic scalar expression preserved from runtime scalar construction. -/
inductive ScalarExpr where
  | var (v : VarId)
  | param (v : VarId) (name : String) (ty : String)
  | const (value : Int)
  | threadIdx (axis : Nat)
  | blockIdx (axis : Nat)
  | gridDim (axis : Nat)
  | warpGroupIdx
  | layoutDim (src : VarId) (axis : LayoutDimAxis)
  | unary (op : ScalarUnaryOp) (src : ScalarExpr)
  | binary (op : ScalarBinaryOp) (lhs rhs : ScalarExpr)
  | select (cond : ThreadPred) (ifTrue ifFalse : ScalarExpr)
  deriving Repr, Inhabited, BEq

/-- Structured thread predicate for guards and proof obligations. -/
inductive ThreadPred where
  | top
  | bottom
  | boolVar (v : VarId)
  | not (p : ThreadPred)
  | and (lhs rhs : ThreadPred)
  | or (lhs rhs : ThreadPred)
  | cmp (op : ScalarCompareOp) (lhs rhs : ScalarExpr)
  | warpGroupEq (idx : Nat)
  deriving Repr, Inhabited, BEq

end

namespace ScalarExpr

def add (a b : ScalarExpr) : ScalarExpr := .binary .Add a b
def sub (a b : ScalarExpr) : ScalarExpr := .binary .Sub a b
def mul (a b : ScalarExpr) : ScalarExpr := .binary .Mul a b
def div (a b : ScalarExpr) : ScalarExpr := .binary .Div a b
def mod (a b : ScalarExpr) : ScalarExpr := .binary .Mod a b

end ScalarExpr

namespace ThreadPred

def andMany (preds : Array ThreadPred) : ThreadPred :=
  preds.foldl .and .top

def implies (lhs rhs : ThreadPred) : ThreadPred :=
  .or (.not lhs) rhs

private def evalCmp (op : ScalarCompareOp) (lhs rhs : Int) : Bool :=
  match op with
  | .Eq => lhs == rhs
  | .Lt => lhs < rhs
  | .Le => lhs <= rhs
  | .Gt => lhs > rhs
  | .Ge => lhs >= rhs

end ThreadPred

mutual

partial def ScalarExpr.render : ScalarExpr -> String
  | .var v => v.toIdent
  | .param v name _ => if name.isEmpty then v.toIdent else name
  | .const value => toString value
  | .threadIdx 0 => "threadIdx.x"
  | .threadIdx 1 => "threadIdx.y"
  | .threadIdx _ => "threadIdx.z"
  | .blockIdx 0 => "blockIdx.x"
  | .blockIdx 1 => "blockIdx.y"
  | .blockIdx _ => "blockIdx.z"
  | .gridDim 0 => "gridDim.x"
  | .gridDim 1 => "gridDim.y"
  | .gridDim _ => "gridDim.z"
  | .warpGroupIdx => "warpGroupIdx"
  | .layoutDim src .Batch => s!"{src.toIdent}.batch"
  | .layoutDim src .Depth => s!"{src.toIdent}.depth"
  | .layoutDim src .Rows => s!"{src.toIdent}.rows"
  | .layoutDim src .Cols => s!"{src.toIdent}.cols"
  | .unary .Neg src => s!"(-{src.render})"
  | .unary .Exp src => s!"exp({src.render})"
  | .binary op lhs rhs =>
      let opStr := match op with
        | .Add => "+"
        | .Sub => "-"
        | .Mul => "*"
        | .Div => "/"
        | .Mod => "%"
        | .Min => "min"
        | .Max => "max"
      match op with
      | .Min | .Max => s!"{opStr}({lhs.render}, {rhs.render})"
      | _ => s!"({lhs.render} {opStr} {rhs.render})"
  | .select cond t f => s!"({cond.render} ? {t.render} : {f.render})"

partial def ThreadPred.render : ThreadPred -> String
  | .top => "true"
  | .bottom => "false"
  | .boolVar v => v.toIdent
  | .not p => s!"!({p.render})"
  | .and lhs rhs => s!"({lhs.render} && {rhs.render})"
  | .or lhs rhs => s!"({lhs.render} || {rhs.render})"
  | .cmp op lhs rhs =>
      let opStr := match op with
        | .Eq => "=="
        | .Lt => "<"
        | .Le => "<="
        | .Gt => ">"
        | .Ge => ">="
      s!"({lhs.render} {opStr} {rhs.render})"
  | .warpGroupEq idx => s!"(warpGroupIdx == {idx})"

end

mutual

def ScalarExpr.eval? (ctx : ThreadCtx) (tx ty tz : Nat) : ScalarExpr -> Option Int
  | .const value => some value
  | .threadIdx axis => some (Int.ofNat <| match axis with
      | 0 => tx
      | 1 => ty
      | _ => tz)
  | .warpGroupIdx => some (Int.ofNat <| ctx.warpGroupOf tx ty tz)
  | .unary .Neg src => src.eval? ctx tx ty tz |>.map (fun x => -x)
  | .binary op lhs rhs => do
      let a <- lhs.eval? ctx tx ty tz
      let b <- rhs.eval? ctx tx ty tz
      match op with
      | .Add => some (a + b)
      | .Sub => some (a - b)
      | .Mul => some (a * b)
      | .Div => if b == 0 then none else some (a / b)
      | .Mod => if b == 0 then none else some (a % b)
      | .Min => some (min a b)
      | .Max => some (max a b)
  | .select cond t f => do
      if <- cond.eval? ctx tx ty tz then
        t.eval? ctx tx ty tz
      else
        f.eval? ctx tx ty tz
  | _ => none

def ThreadPred.eval? (ctx : ThreadCtx) (tx ty tz : Nat) : ThreadPred -> Option Bool
  | .top => some true
  | .bottom => some false
  | .boolVar _ => none
  | .not p => p.eval? ctx tx ty tz |>.map (fun b => !b)
  | .and lhs rhs => do
      let a <- lhs.eval? ctx tx ty tz
      let b <- rhs.eval? ctx tx ty tz
      some (a && b)
  | .or lhs rhs => do
      let a <- lhs.eval? ctx tx ty tz
      let b <- rhs.eval? ctx tx ty tz
      some (a || b)
  | .cmp op lhs rhs => do
      let a <- lhs.eval? ctx tx ty tz
      let b <- rhs.eval? ctx tx ty tz
      some (ThreadPred.evalCmp op a b)
  | .warpGroupEq idx => some (ctx.warpGroupOf tx ty tz == idx)

end

namespace ThreadPred

partial def mentionsUnknownBool : ThreadPred -> Bool
  | .boolVar _ => true
  | .not p => p.mentionsUnknownBool
  | .and lhs rhs | .or lhs rhs => lhs.mentionsUnknownBool || rhs.mentionsUnknownBool
  | _ => false

/--
Conservative structural truth under an already-active guard.

This is intentionally not a diagnostic evaluator: it only recognizes identities
that are true by the predicate syntax itself and is used by proof-aware IR
rewrites to avoid introducing string/native-Boolean proof paths.
-/
def truthUnder? (guard pred : ThreadPred) : Option Bool :=
  match pred with
  | .top => some true
  | .bottom => some false
  | _ =>
      if pred == guard then some true
      else if pred == .not guard then some false
      else none

end ThreadPred

end Tyr.GPU.Codegen
