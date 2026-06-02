import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.Stmt
import Tyr.ShapeSync

/-!
# Tyr.GPU.Codegen.Proof

Lean-native proof metadata for GPU kernel safety and optimization checks.

This module defines the symbolic language, thread model, obligations, and
optimization certificates. Where possible it derives proof facts directly from
`KStmt` so proof metadata tracks the lowered IR instead of a parallel string
description. Finite evaluators in this module are diagnostic aids only; accepted
proof certificates must prove the semantic `Holds` proposition.
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

def axisBound (ctx : ThreadCtx) : Nat → Nat
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

namespace ShapeSyncBridge

def threadCtx (ctx : ThreadCtx) : Tyr.ShapeSync.ThreadCtx :=
  {
    blockDimX := ctx.blockDimX
    blockDimY := ctx.blockDimY
    blockDimZ := ctx.blockDimZ
    warpSize := ctx.warpSize
    warpGroupSize := ctx.warpGroupSize
  }

private def natConst? : ScalarExpr → Option Nat
  | .const value => if value < 0 then none else some value.toNat
  | _ => none

private def positiveNatConst? (expr : ScalarExpr) : Option Nat := do
  let value ← natConst? expr
  if value == 0 then none else some value

private def eqPred? (lhs rhs : ScalarExpr) :
    Option Tyr.ShapeSync.ThreadPred :=
  match lhs, rhs with
  | .threadIdx axis, _ => do
      let value ← natConst? rhs
      some (.axisEq axis value)
  | _, .threadIdx axis => do
      let value ← natConst? lhs
      some (.axisEq axis value)
  | .warpGroupIdx, _ => do
      let value ← natConst? rhs
      some (.warpGroupEq value)
  | _, .warpGroupIdx => do
      let value ← natConst? lhs
      some (.warpGroupEq value)
  | .binary .Mod (.threadIdx axis) modulus, _ => do
      let modulus ← positiveNatConst? modulus
      let residue ← natConst? rhs
      some (.modEq axis modulus residue)
  | _, .binary .Mod (.threadIdx axis) modulus => do
      let modulus ← positiveNatConst? modulus
      let residue ← natConst? lhs
      some (.modEq axis modulus residue)
  | _, _ => none

private def ltPred? (lhs rhs : ScalarExpr) :
    Option Tyr.ShapeSync.ThreadPred :=
  match lhs with
  | .threadIdx axis => do
      let upper ← natConst? rhs
      some (.axisLt axis upper)
  | _ => none

private def cmpPred? (op : ScalarCompareOp) (lhs rhs : ScalarExpr) :
    Option Tyr.ShapeSync.ThreadPred :=
  match op with
  | .Eq => eqPred? lhs rhs
  | .Lt => ltPred? lhs rhs
  | .Le => do
      match lhs with
      | .threadIdx axis => do
          let upper ← natConst? rhs
          some (.axisLt axis (upper + 1))
      | _ => none
  | .Gt => ltPred? rhs lhs
  | .Ge => do
      match rhs with
      | .threadIdx axis => do
          let upper ← natConst? lhs
          some (.axisLt axis (upper + 1))
      | _ => none

partial def threadPred? : ThreadPred → Option Tyr.ShapeSync.ThreadPred
  | .top => some .top
  | .bottom => some .bottom
  | .boolVar _ => none
  | .not pred => do
      some (.not (← threadPred? pred))
  | .and lhs rhs => do
      some (.and (← threadPred? lhs) (← threadPred? rhs))
  | .or lhs rhs => do
      some (.or (← threadPred? lhs) (← threadPred? rhs))
  | .cmp op lhs rhs => cmpPred? op lhs rhs
  | .warpGroupEq idx => some (.warpGroupEq idx)

end ShapeSyncBridge

mutual

partial def ScalarExpr.render : ScalarExpr → String
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

partial def ThreadPred.render : ThreadPred → String
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

def ScalarExpr.eval? (ctx : ThreadCtx) (tx ty tz : Nat) : ScalarExpr → Option Int
  | .const value => some value
  | .threadIdx axis => some (Int.ofNat <| match axis with
      | 0 => tx
      | 1 => ty
      | _ => tz)
  | .warpGroupIdx => some (Int.ofNat <| ctx.warpGroupOf tx ty tz)
  | .unary .Neg src => src.eval? ctx tx ty tz |>.map (fun x => -x)
  | .binary op lhs rhs => do
      let a ← lhs.eval? ctx tx ty tz
      let b ← rhs.eval? ctx tx ty tz
      match op with
      | .Add => some (a + b)
      | .Sub => some (a - b)
      | .Mul => some (a * b)
      | .Div => if b == 0 then none else some (a / b)
      | .Mod => if b == 0 then none else some (a % b)
      | .Min => some (min a b)
      | .Max => some (max a b)
  | .select cond t f => do
      if ← cond.eval? ctx tx ty tz then
        t.eval? ctx tx ty tz
      else
        f.eval? ctx tx ty tz
  | _ => none

def ThreadPred.eval? (ctx : ThreadCtx) (tx ty tz : Nat) : ThreadPred → Option Bool
  | .top => some true
  | .bottom => some false
  | .boolVar _ => none
  | .not p => p.eval? ctx tx ty tz |>.map (fun b => !b)
  | .and lhs rhs => do
      let a ← lhs.eval? ctx tx ty tz
      let b ← rhs.eval? ctx tx ty tz
      some (a && b)
  | .or lhs rhs => do
      let a ← lhs.eval? ctx tx ty tz
      let b ← rhs.eval? ctx tx ty tz
      some (a || b)
  | .cmp op lhs rhs => do
      let a ← lhs.eval? ctx tx ty tz
      let b ← rhs.eval? ctx tx ty tz
      some (ThreadPred.evalCmp op a b)
  | .warpGroupEq idx => some (ctx.warpGroupOf tx ty tz == idx)

end

namespace ThreadPred

partial def mentionsUnknownBool : ThreadPred → Bool
  | .boolVar _ => true
  | .not p => p.mentionsUnknownBool
  | .and lhs rhs | .or lhs rhs => lhs.mentionsUnknownBool || rhs.mentionsUnknownBool
  | _ => false

end ThreadPred

/-- Closed integer interval. `hi < lo` represents an empty interval. -/
structure IntInterval where
  lo : Int
  hi : Int
  deriving Repr, Inhabited, BEq

namespace IntInterval

def empty : IntInterval := { lo := 1, hi := 0 }
def singleton (x : Int) : IntInterval := { lo := x, hi := x }
def isEmpty (i : IntInterval) : Bool := i.hi < i.lo
def disjoint (a b : IntInterval) : Bool :=
  a.isEmpty || b.isEmpty || a.hi < b.lo || b.hi < a.lo

def extendByLength (i : IntInterval) (len : Nat) : IntInterval :=
  if i.isEmpty then i
  else { i with hi := i.hi + Int.ofNat (len - 1) }

def insert (i : IntInterval) (x : Int) : IntInterval :=
  if i.isEmpty then singleton x
  else { lo := min i.lo x, hi := max i.hi x }

end IntInterval

/-- Approximate the values of an expression over all threads satisfying a guard. -/
def exprInterval? (ctx : ThreadCtx) (guard : ThreadPred) (expr : ScalarExpr) :
    Option IntInterval := Id.run do
  let mut interval := IntInterval.empty
  for tx in [:ctx.blockDimX] do
    for ty in [:ctx.blockDimY] do
      for tz in [:ctx.blockDimZ] do
        match guard.eval? ctx tx ty tz with
        | none => return none
        | some false => pure ()
        | some true =>
            match expr.eval? ctx tx ty tz with
            | none => return none
            | some value => interval := interval.insert value
  return some interval

/-- Count participating threads for a structured guard. -/
def participantCount? (ctx : ThreadCtx) (guard : ThreadPred) : Option Nat := Id.run do
  let mut count := 0
  for tx in [:ctx.blockDimX] do
    for ty in [:ctx.blockDimY] do
      for tz in [:ctx.blockDimZ] do
        match guard.eval? ctx tx ty tz with
        | none => return none
        | some true => count := count + 1
        | some false => pure ()
  return some count

/-- Decide whether two guards can ever be true for the same thread. -/
def guardsOverlap? (ctx : ThreadCtx) (a b : ThreadPred) : Option Bool := Id.run do
  for tx in [:ctx.blockDimX] do
    for ty in [:ctx.blockDimY] do
      for tz in [:ctx.blockDimZ] do
        match a.eval? ctx tx ty tz, b.eval? ctx tx ty tz with
        | some av, some bv =>
            if av && bv then
              return some true
        | _, _ => return none
  return some false

def guardImplies? (ctx : ThreadCtx) (lhs rhs : ThreadPred) : Option Bool := Id.run do
  for tx in [:ctx.blockDimX] do
    for ty in [:ctx.blockDimY] do
      for tz in [:ctx.blockDimZ] do
        match lhs.eval? ctx tx ty tz, rhs.eval? ctx tx ty tz with
        | some true, some false => return some false
        | some _, some _ => pure ()
        | _, _ => return none
  return some true

def guardsEquivalent? (ctx : ThreadCtx) (lhs rhs : ThreadPred) : Option Bool := do
  let l ← guardImplies? ctx lhs rhs
  let r ← guardImplies? ctx rhs lhs
  some (l && r)

def exprsEqualUnderGuard? (ctx : ThreadCtx) (guard : ThreadPred)
    (lhs rhs : ScalarExpr) : Option Bool := Id.run do
  for tx in [:ctx.blockDimX] do
    for ty in [:ctx.blockDimY] do
      for tz in [:ctx.blockDimZ] do
        match guard.eval? ctx tx ty tz with
        | some false => pure ()
        | some true =>
            match lhs.eval? ctx tx ty tz, rhs.eval? ctx tx ty tz with
            | some a, some b =>
                if a != b then
                  return some false
            | _, _ => return none
        | none => return none
  return some true

def exprDivisibleUnderGuard? (ctx : ThreadCtx) (guard : ThreadPred)
    (expr : ScalarExpr) (d : Nat) : Option Bool := Id.run do
  if d == 0 then
    return some false
  let divisor := Int.ofNat d
  for tx in [:ctx.blockDimX] do
    for ty in [:ctx.blockDimY] do
      for tz in [:ctx.blockDimZ] do
        match guard.eval? ctx tx ty tz with
        | some false => pure ()
        | some true =>
            match expr.eval? ctx tx ty tz with
            | some value =>
                if value % divisor != 0 then
                  return some false
            | none => return none
        | none => return none
  return some true

def exprNonNegativeUnderGuard? (ctx : ThreadCtx) (guard : ThreadPred)
    (expr : ScalarExpr) : Option Bool := Id.run do
  for tx in [:ctx.blockDimX] do
    for ty in [:ctx.blockDimY] do
      for tz in [:ctx.blockDimZ] do
        match guard.eval? ctx tx ty tz with
        | some false => pure ()
        | some true =>
            match expr.eval? ctx tx ty tz with
            | some value =>
                if value < 0 then
                  return some false
            | none => return none
        | none => return none
  return some true

def predicateValueUnderGuard? (ctx : ThreadCtx) (guard pred : ThreadPred)
    (expected : Bool) : Option Bool := Id.run do
  for tx in [:ctx.blockDimX] do
    for ty in [:ctx.blockDimY] do
      for tz in [:ctx.blockDimZ] do
        match guard.eval? ctx tx ty tz with
        | some false => pure ()
        | some true =>
            match pred.eval? ctx tx ty tz with
            | some actual =>
                if actual != expected then
                  return some false
            | none => return none
        | none => return none
  return some true

/-- Semantic source of a memory access tracked for race/disjointness checks. -/
inductive AccessKind where
  | loadScalarGlobal
  | storeScalarGlobal
  | loadVecGlobal
  | storeVecGlobal
  | storeVecGlobalAdd
  deriving Repr, Inhabited, BEq

namespace AccessKind

def render : AccessKind → String
  | .loadScalarGlobal => "loadScalarGlobal"
  | .storeScalarGlobal => "storeScalarGlobal"
  | .loadVecGlobal => "loadVecGlobal"
  | .storeVecGlobal => "storeVecGlobal"
  | .storeVecGlobalAdd => "storeVecGlobalAdd"

end AccessKind

/-- Memory address range touched by a guarded access. -/
structure AccessRange where
  base : VarId
  offset : ScalarExpr
  width : Nat := 1
  guard : ThreadPred := .top
  isWrite : Bool := false
  isAtomic : Bool := false
  kind : AccessKind := .loadScalarGlobal
  deriving Repr, Inhabited, BEq

namespace AccessRange

def render (r : AccessRange) : String :=
  let mode := if r.isAtomic then "atomic" else if r.isWrite then "write" else "read"
  s!"{mode} {r.base.toIdent}[{r.offset.render} .. +{r.width}] when {r.guard.render}"

def interval? (ctx : ThreadCtx) (r : AccessRange) : Option IntInterval := do
  let i ← exprInterval? ctx r.guard r.offset
  some (i.extendByLength r.width)

def disjoint? (ctx : ThreadCtx) (a b : AccessRange) : Option Bool := do
  if a.base != b.base then
    return true
  match ← guardsOverlap? ctx a.guard b.guard with
  | false => return true
  | true =>
      let ai ← a.interval? ctx
      let bi ← b.interval? ctx
      return ai.disjoint bi

end AccessRange

/-- Semantic facts about values declared in the kernel IR. -/
inductive KVarSemantics where
  | scalar
  | globalPointer (dtype : GpuFloat)
  | registerTile (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  | sharedTile (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  | registerVector (dtype : GpuFloat) (len : Nat)
  | sharedVector (dtype : GpuFloat) (len : Nat)
  | semaphore
  | tensorTile (dtype : GpuFloat) (rows cols : Nat)
  | tensorMemoryPool (slots clusterSize : Nat) (managed : Bool)
  deriving Repr, Inhabited, BEq

namespace KVarSemantics

def elementCount : KVarSemantics → Nat
  | .registerTile _ rows cols _ => rows * cols
  | .sharedTile _ rows cols _ => rows * cols
  | .registerVector _ len => len
  | .sharedVector _ len => len
  | .tensorTile _ rows cols => rows * cols
  | _ => 1

end KVarSemantics

inductive ProofBackend where
  | grind
  | bvDecide
  deriving Repr, Inhabited, BEq

namespace ProofBackend

def tacticName : ProofBackend → String
  | .grind => "grind"
  | .bvDecide => "bv_decide"

end ProofBackend

/-- The synchronization statement that introduced a barrier proof obligation. -/
inductive BarrierSiteKind where
  | namedBarrierSync
  | namedBarrierArrive
  deriving Repr, Inhabited, BEq

namespace BarrierSiteKind

def render : BarrierSiteKind → String
  | .namedBarrierSync => "namedBarrierSync"
  | .namedBarrierArrive => "namedBarrierArrive"

end BarrierSiteKind

/-- Provenance for an optimization certificate. The claim carries semantics; this is identity. -/
structure OptimizationSite where
  pass : Lean.Name := `kernel
  rule : Lean.Name := `unknown
  deriving Repr, Inhabited, BEq

namespace OptimizationSite

def render (site : OptimizationSite) : String :=
  s!"{site.pass}.{site.rule}"

end OptimizationSite

/-- Typed origin for a proof obligation. Rendering is diagnostic only. -/
inductive ProofSite where
  | kernel
  | barrier (kind : BarrierSiteKind) (id : Nat)
  | accessPair (prior current : AccessKind)
  | optimization (site : OptimizationSite)
  deriving Repr, Inhabited, BEq

namespace ProofSite

def render : ProofSite → String
  | .kernel => "kernel"
  | .barrier kind id => s!"{kind.render}#{id}"
  | .accessPair prior current => s!"{prior.render}/{current.render}"
  | .optimization site => s!"optimization/{site.render}"

end ProofSite

/-- Resource category for static kernel resource obligations. -/
inductive ResourceProperty where
  | sharedMemoryBytes
  | tensorMemorySlots
  | blockThreads
  deriving Repr, Inhabited, BEq

namespace ResourceProperty

def render : ResourceProperty → String
  | .sharedMemoryBytes => "shared_memory_bytes"
  | .tensorMemorySlots => "tensor_memory_slots"
  | .blockThreads => "block_threads"

end ResourceProperty

/-- Shape category for static divisibility obligations. -/
inductive ShapeProperty where
  | tileExtent
  | vectorWidth
  | blockDim
  deriving Repr, Inhabited, BEq

namespace ShapeProperty

def render : ShapeProperty → String
  | .tileExtent => "tile_extent"
  | .vectorWidth => "vector_width"
  | .blockDim => "block_dim"

end ShapeProperty

/-- Safety obligations generated by checked APIs and validation passes. -/
inductive ProofObligationKind where
  | implication (lhs rhs : ThreadPred)
  | equivalence (lhs rhs : ThreadPred)
  | scalarEquality (guard : ThreadPred) (lhs rhs : ScalarExpr)
  | scalarDivisible (guard : ThreadPred) (expr : ScalarExpr) (d : Nat)
  | scalarNonNegative (guard : ThreadPred) (expr : ScalarExpr)
  | predicateValue (guard pred : ThreadPred) (expected : Bool)
  | participantCount (guard : ThreadPred) (expected : Nat)
  | disjoint (lhs rhs : AccessRange)
  | resourceLimit (property : ResourceProperty) (actual limit : Nat)
  | natDivisible (property : ShapeProperty) (n d : Nat)
  deriving Repr, Inhabited, BEq

namespace ProofObligationKind

def backend : ProofObligationKind → ProofBackend
  | .implication .. | .equivalence .. | .scalarEquality ..
  | .scalarDivisible .. | .scalarNonNegative .. | .predicateValue ..
  | .participantCount .. => .grind
  | .disjoint .. | .resourceLimit .. | .natDivisible .. => .grind

def guard : ProofObligationKind → ThreadPred
  | .implication lhs _ => lhs
  | .equivalence lhs _ => lhs
  | .scalarEquality guard .. => guard
  | .scalarDivisible guard .. => guard
  | .scalarNonNegative guard .. => guard
  | .predicateValue guard .. => guard
  | .participantCount guard _ => guard
  | .disjoint lhs rhs => .and lhs.guard rhs.guard
  | .resourceLimit .. | .natDivisible .. => .top

def render : ProofObligationKind → String
  | .implication lhs rhs => s!"{lhs.render} -> {rhs.render}"
  | .equivalence lhs rhs => s!"{lhs.render} <-> {rhs.render}"
  | .scalarEquality guard lhs rhs =>
      s!"under {guard.render}: {lhs.render} = {rhs.render}"
  | .scalarDivisible guard expr d =>
      s!"under {guard.render}: {expr.render} % {d} = 0"
  | .scalarNonNegative guard expr =>
      s!"under {guard.render}: 0 <= {expr.render}"
  | .predicateValue guard pred expected =>
      let rhs := if expected then "true" else "false"
      s!"under {guard.render}: {pred.render} = {rhs}"
  | .participantCount guard expected =>
      s!"participant_count({guard.render}) = {expected}"
  | .disjoint lhs rhs => s!"disjoint({lhs.render}, {rhs.render})"
  | .resourceLimit property actual limit =>
      s!"{property.render}: {actual} <= {limit}"
  | .natDivisible property n d => s!"{property.render}: {n} % {d} = 0"

/-- Lean-native meaning of a proof obligation, independent of diagnostics. -/
def Holds (ctx : ThreadCtx) : ProofObligationKind → Prop
  | .implication lhs rhs =>
      ∀ tx ty tz,
        tx < ctx.blockDimX → ty < ctx.blockDimY → tz < ctx.blockDimZ →
        lhs.eval? ctx tx ty tz = some true →
        rhs.eval? ctx tx ty tz = some true
  | .equivalence lhs rhs =>
      ∀ tx ty tz,
        tx < ctx.blockDimX → ty < ctx.blockDimY → tz < ctx.blockDimZ →
        lhs.eval? ctx tx ty tz = rhs.eval? ctx tx ty tz
  | .scalarEquality guard lhs rhs =>
      ∀ tx ty tz,
        tx < ctx.blockDimX → ty < ctx.blockDimY → tz < ctx.blockDimZ →
        guard.eval? ctx tx ty tz = some true →
        ∃ value, lhs.eval? ctx tx ty tz = some value ∧
          rhs.eval? ctx tx ty tz = some value
  | .scalarDivisible guard expr d =>
      d != 0 ∧
      ∀ tx ty tz,
        tx < ctx.blockDimX → ty < ctx.blockDimY → tz < ctx.blockDimZ →
        guard.eval? ctx tx ty tz = some true →
        ∃ value, expr.eval? ctx tx ty tz = some value ∧ value % Int.ofNat d = 0
  | .scalarNonNegative guard expr =>
      ∀ tx ty tz,
        tx < ctx.blockDimX → ty < ctx.blockDimY → tz < ctx.blockDimZ →
        guard.eval? ctx tx ty tz = some true →
        ∃ value, expr.eval? ctx tx ty tz = some value ∧ 0 <= value
  | .predicateValue guard pred expected =>
      ∀ tx ty tz,
        tx < ctx.blockDimX → ty < ctx.blockDimY → tz < ctx.blockDimZ →
        guard.eval? ctx tx ty tz = some true →
        pred.eval? ctx tx ty tz = some expected
  | .participantCount guard expected =>
      participantCount? ctx guard = some expected
  | .disjoint lhs rhs =>
      AccessRange.disjoint? ctx lhs rhs = some true
  | .resourceLimit _ actual limit =>
      actual <= limit
  | .natDivisible _ n d =>
      d != 0 ∧ n % d = 0

end ProofObligationKind

structure ProofObligation where
  kind : ProofObligationKind
  site : ProofSite := .kernel
  deriving Repr, Inhabited, BEq

namespace ProofObligation

def render (obl : ProofObligation) : String :=
  s!"{obl.site.render}: {obl.kind.render}"

end ProofObligation

inductive ProofFailureReason where
  | false
  | unsupported
  deriving Repr, Inhabited, BEq

namespace ProofFailureReason

def render : ProofFailureReason → String
  | .false => "proof obligation is false"
  | .unsupported => "unsupported proof shape"

end ProofFailureReason

structure ProofFailure where
  kernelName : String
  site : ProofSite
  guard : ThreadPred
  proposition : ProofObligationKind
  reason : ProofFailureReason
  deriving Repr, Inhabited, BEq

namespace ProofFailure

def render (failure : ProofFailure) : String :=
  s!"{failure.kernelName}: {failure.site.render} under {failure.guard.render}: " ++
  s!"{failure.reason.render}; obligation: {failure.proposition.render}"

end ProofFailure

/- Best-effort finite/interval diagnostics. These checks never certify an obligation. -/
namespace ProofDiagnostics

def check? (ctx : ThreadCtx) : ProofObligationKind → Option Bool
  | .implication lhs rhs => guardImplies? ctx lhs rhs
  | .equivalence lhs rhs => guardsEquivalent? ctx lhs rhs
  | .scalarEquality guard lhs rhs => exprsEqualUnderGuard? ctx guard lhs rhs
  | .scalarDivisible guard expr d => exprDivisibleUnderGuard? ctx guard expr d
  | .scalarNonNegative guard expr => exprNonNegativeUnderGuard? ctx guard expr
  | .predicateValue guard pred expected => predicateValueUnderGuard? ctx guard pred expected
  | .participantCount guard expected => do
      let actual ← participantCount? ctx guard
      some (actual == expected)
  | .disjoint lhs rhs => AccessRange.disjoint? ctx lhs rhs
  | .resourceLimit _ actual limit => some (actual <= limit)
  | .natDivisible _ n d => some (d != 0 && n % d == 0)

end ProofDiagnostics

/-- A validated obligation carries a Lean proof of the obligation semantics. -/
structure ValidatedObligation (ctx : ThreadCtx) where
  obligation : ProofObligation
  certificate : obligation.kind.Holds ctx

namespace ValidatedObligation

def backend (checked : ValidatedObligation ctx) : ProofBackend :=
  checked.obligation.kind.backend

def render (checked : ValidatedObligation ctx) : String :=
  checked.obligation.render

end ValidatedObligation

namespace ProofObligation

def certify (ctx : ThreadCtx) (obl : ProofObligation)
    (proof : obl.kind.Holds ctx) : ValidatedObligation ctx :=
  { obligation := obl, certificate := proof }

end ProofObligation

inductive ProofResult (ctx : ThreadCtx) where
  | proved (certificate : ValidatedObligation ctx)
  | rejected (failure : ProofFailure)

namespace ProofResult

def isProved : ProofResult ctx → Bool
  | .proved _ => true
  | .rejected _ => false

end ProofResult

/-- Typed optimization claim made by a rewrite or lowering pass. -/
inductive OptimizationClaim where
  | scalarRewrite (guard : ThreadPred) (before after : ScalarExpr)
  | predicateRewrite (before after : ThreadPred)
  | predicateSimplification (guard pred : ThreadPred) (expected : Bool)
  | alignmentCheck (guard : ThreadPred) (expr : ScalarExpr) (d : Nat)
  | syncOmission (guard dependencySatisfied : ThreadPred)
  | asyncWaitOmission (guard waitRedundant : ThreadPred)
  deriving Repr, Inhabited, BEq

namespace OptimizationClaim

def render : OptimizationClaim → String
  | .scalarRewrite _ before after =>
      s!"{before.render} => {after.render}"
  | .predicateRewrite before after =>
      s!"{before.render} => {after.render}"
  | .predicateSimplification _ pred expected =>
      let rhs := if expected then "true" else "false"
      s!"{pred.render} => {rhs}"
  | .alignmentCheck _ expr d =>
      s!"aligned({expr.render}, {d})"
  | .syncOmission _ dependencySatisfied =>
      s!"omit_sync when {dependencySatisfied.render}"
  | .asyncWaitOmission _ waitRedundant =>
      s!"omit_async_wait when {waitRedundant.render}"

def obligation (site : OptimizationSite) : OptimizationClaim → ProofObligation
  | .scalarRewrite guard before after =>
      {
        kind := .scalarEquality guard before after
        site := .optimization site
      }
  | .predicateRewrite before after =>
      {
        kind := .equivalence before after
        site := .optimization site
      }
  | .predicateSimplification guard pred expected =>
      {
        kind := .predicateValue guard pred expected
        site := .optimization site
      }
  | .alignmentCheck guard expr d =>
      {
        kind := .scalarDivisible guard expr d
        site := .optimization site
      }
  | .syncOmission guard dependencySatisfied =>
      {
        kind := .predicateValue guard dependencySatisfied true
        site := .optimization site
      }
  | .asyncWaitOmission guard waitRedundant =>
      {
        kind := .predicateValue guard waitRedundant true
        site := .optimization site
      }

def obligations (site : OptimizationSite) (claim : OptimizationClaim) : Array ProofObligation :=
  #[claim.obligation site]

end OptimizationClaim

/-- A proof-backed optimization decision made by a rewrite or lowering pass. -/
structure OptimizationProof where
  site : OptimizationSite
  claim : OptimizationClaim
  deriving Repr, Inhabited, BEq

namespace OptimizationProof

def render (opt : OptimizationProof) : String :=
  s!"{opt.site.render}: {opt.claim.render}"

def obligations (opt : OptimizationProof) : Array ProofObligation :=
  opt.claim.obligations opt.site

def scalarRewrite (passName rewriteName : Lean.Name) (guard : ThreadPred)
    (before after : ScalarExpr) : OptimizationProof :=
  { site := { pass := passName, rule := rewriteName }
    claim := .scalarRewrite guard before after }

def predicateRewrite (passName rewriteName : Lean.Name)
    (before after : ThreadPred) : OptimizationProof :=
  { site := { pass := passName, rule := rewriteName }
    claim := .predicateRewrite before after }

def predicateSimplification (passName rewriteName : Lean.Name) (guard pred : ThreadPred)
    (expected : Bool) : OptimizationProof :=
  { site := { pass := passName, rule := rewriteName }
    claim := .predicateSimplification guard pred expected }

def alignmentCheck (passName rewriteName : Lean.Name) (guard : ThreadPred)
    (expr : ScalarExpr) (d : Nat) : OptimizationProof :=
  { site := { pass := passName, rule := rewriteName }
    claim := .alignmentCheck guard expr d }

def syncOmission (passName rewriteName : Lean.Name)
    (guard dependencySatisfied : ThreadPred) : OptimizationProof :=
  { site := { pass := passName, rule := rewriteName }
    claim := .syncOmission guard dependencySatisfied }

def asyncWaitOmission (passName rewriteName : Lean.Name)
    (guard waitRedundant : ThreadPred) : OptimizationProof :=
  { site := { pass := passName, rule := rewriteName }
    claim := .asyncWaitOmission guard waitRedundant }

end OptimizationProof

/-- An optimization decision accompanied by a proof of its enabling obligation. -/
structure CertifiedOptimization (ctx : ThreadCtx) where
  optimization : OptimizationProof
  certificate : ValidatedObligation ctx

namespace CertifiedOptimization

def ofProof (ctx : ThreadCtx) (opt : OptimizationProof)
    (proof : (opt.claim.obligation opt.site).kind.Holds ctx) :
    CertifiedOptimization ctx :=
  { optimization := opt
    certificate := (opt.claim.obligation opt.site).certify ctx proof }

def scalarRewrite (ctx : ThreadCtx) (passName rewriteName : Lean.Name)
    (guard : ThreadPred) (before after : ScalarExpr)
    (proof : (ProofObligationKind.scalarEquality guard before after).Holds ctx) :
    CertifiedOptimization ctx :=
  ofProof ctx (OptimizationProof.scalarRewrite passName rewriteName guard before after) proof

def predicateRewrite (ctx : ThreadCtx) (passName rewriteName : Lean.Name)
    (before after : ThreadPred)
    (proof : (ProofObligationKind.equivalence before after).Holds ctx) :
    CertifiedOptimization ctx :=
  ofProof ctx (OptimizationProof.predicateRewrite passName rewriteName before after) proof

def predicateSimplification (ctx : ThreadCtx) (passName rewriteName : Lean.Name)
    (guard pred : ThreadPred) (expected : Bool)
    (proof : (ProofObligationKind.predicateValue guard pred expected).Holds ctx) :
    CertifiedOptimization ctx :=
  ofProof ctx (OptimizationProof.predicateSimplification passName rewriteName guard pred expected) proof

def alignmentCheck (ctx : ThreadCtx) (passName rewriteName : Lean.Name)
    (guard : ThreadPred) (expr : ScalarExpr) (d : Nat)
    (proof : (ProofObligationKind.scalarDivisible guard expr d).Holds ctx) :
    CertifiedOptimization ctx :=
  ofProof ctx (OptimizationProof.alignmentCheck passName rewriteName guard expr d) proof

def syncOmission (ctx : ThreadCtx) (passName rewriteName : Lean.Name)
    (guard dependencySatisfied : ThreadPred)
    (proof : (ProofObligationKind.predicateValue guard dependencySatisfied true).Holds ctx) :
    CertifiedOptimization ctx :=
  ofProof ctx (OptimizationProof.syncOmission passName rewriteName guard dependencySatisfied) proof

def asyncWaitOmission (ctx : ThreadCtx) (passName rewriteName : Lean.Name)
    (guard waitRedundant : ThreadPred)
    (proof : (ProofObligationKind.predicateValue guard waitRedundant true).Holds ctx) :
    CertifiedOptimization ctx :=
  ofProof ctx (OptimizationProof.asyncWaitOmission passName rewriteName guard waitRedundant) proof

end CertifiedOptimization

inductive OptimizationDecision (ctx : ThreadCtx) (α : Type u) where
  | accepted (result : α) (certificate : CertifiedOptimization ctx)
  | rejected (failure : ProofFailure)

namespace OptimizationDecision

def isAccepted : OptimizationDecision ctx α → Bool
  | .accepted .. => true
  | .rejected _ => false

end OptimizationDecision

structure Simplified (ctx : ThreadCtx) where
  before : ScalarExpr
  after : ScalarExpr
  certificate : ValidatedObligation ctx

namespace ProofOracle

def proveObligation (ctx : ThreadCtx) (obl : ProofObligation)
    (proof : obl.kind.Holds ctx) : ProofResult ctx :=
  .proved (obl.certify ctx proof)

def rejectObligation (kernelName : String) (ctx : ThreadCtx)
    (obl : ProofObligation) : ProofResult ctx :=
  let reason :=
    match ProofDiagnostics.check? ctx obl.kind with
    | some false => .false
    | _ => .unsupported
  .rejected {
    kernelName := kernelName
    site := obl.site
    guard := obl.kind.guard
    proposition := obl.kind
    reason := reason
  }

def canProve (ctx : ThreadCtx) (site : ProofSite) (pred : ThreadPred)
    (proof : (ProofObligationKind.predicateValue .top pred true).Holds ctx) :
    ProofResult ctx :=
  proveObligation ctx { kind := .predicateValue .top pred true, site := site } proof

def proveUnder (ctx : ThreadCtx) (site : ProofSite) (guard goal : ThreadPred)
    (proof : (ProofObligationKind.predicateValue guard goal true).Holds ctx) :
    ProofResult ctx :=
  proveObligation ctx { kind := .predicateValue guard goal true, site := site } proof

def proveEqual (ctx : ThreadCtx) (site : ProofSite) (guard : ThreadPred)
    (before after : ScalarExpr)
    (proof : (ProofObligationKind.scalarEquality guard before after).Holds ctx) :
    ProofResult ctx :=
  proveObligation ctx { kind := .scalarEquality guard before after, site := site } proof

def simplifyWithProof (ctx : ThreadCtx) (site : ProofSite) (guard : ThreadPred)
    (before after : ScalarExpr)
    (proof : (ProofObligationKind.scalarEquality guard before after).Holds ctx) :
    Simplified ctx :=
  { before := before
    after := after
    certificate := {
      obligation := { kind := .scalarEquality guard before after, site := site }
      certificate := proof
    } }

end ProofOracle

/-- Proof metadata accumulated while building a kernel. -/
structure KernelProof where
  vars : Array (VarId × KVarSemantics) := #[]
  scalars : Array (VarId × ScalarExpr) := #[]
  predicates : Array (VarId × ThreadPred) := #[]
  accesses : Array AccessRange := #[]
  obligations : Array ProofObligation := #[]
  optimizations : Array OptimizationProof := #[]
  shapeSyncNodes : Array Tyr.ShapeSync.GraphNode := #[]
  shapeSyncGraph? : Option Tyr.ShapeSync.AccessGraph := none
  shapeSyncProducerConsumer? : Option Tyr.ShapeSync.ProducerConsumerAnalysis := none
  shapeSyncPipelineAnalyses : Array Tyr.ShapeSync.ProducerConsumerAnalysis := #[]
  shapeSyncObligations : Array Tyr.ShapeSync.Obligation := #[]
  threadCtx : ThreadCtx := {}
  guardStack : Array ThreadPred := #[]
  deriving Repr, Inhabited, BEq

namespace KernelProof

private def lookupAux [BEq α] (key : α) : Array (α × β) → Option β
  | #[] => none
  | xs =>
      let rec loop (i : Nat) : Option β :=
        if h : i < xs.size then
          let p := xs[i]
          if p.1 == key then some p.2 else loop (i + 1)
        else
          none
      loop 0

private def insertAux [BEq α] (key : α) (value : β) (xs : Array (α × β)) :
    Array (α × β) := Id.run do
  let mut out := #[]
  let mut replaced := false
  for p in xs do
    if p.1 == key then
      out := out.push (key, value)
      replaced := true
    else
      out := out.push p
  if replaced then out else out.push (key, value)

def scalar? (proof : KernelProof) (v : VarId) : Option ScalarExpr :=
  lookupAux v proof.scalars

def var? (proof : KernelProof) (v : VarId) : Option KVarSemantics :=
  lookupAux v proof.vars

def varElementCountD (proof : KernelProof) (v : VarId) (fallback : Nat := 1) : Nat :=
  match proof.var? v with
  | some semantics => semantics.elementCount
  | none => fallback

def scalarOrVar (proof : KernelProof) (v : VarId) : ScalarExpr :=
  proof.scalar? v |>.getD (.var v)

def predicate? (proof : KernelProof) (v : VarId) : Option ThreadPred :=
  lookupAux v proof.predicates

def predicateOrVar (proof : KernelProof) (v : VarId) : ThreadPred :=
  proof.predicate? v |>.getD (.boolVar v)

def insertScalar (proof : KernelProof) (v : VarId) (expr : ScalarExpr) : KernelProof :=
  { proof with scalars := insertAux v expr proof.scalars }

def insertPredicate (proof : KernelProof) (v : VarId) (pred : ThreadPred) : KernelProof :=
  { proof with predicates := insertAux v pred proof.predicates }

def insertVarSemantics (proof : KernelProof) (v : VarId) (semantics : KVarSemantics) :
    KernelProof :=
  { proof with vars := insertAux v semantics proof.vars }

def addObligation (proof : KernelProof) (obl : ProofObligation) : KernelProof :=
  { proof with obligations := proof.obligations.push obl }

private def addOptimizationMetadata (proof : KernelProof) (opt : OptimizationProof) : KernelProof :=
  { proof with
    optimizations := proof.optimizations.push opt
    obligations := proof.obligations ++ opt.obligations
  }

def addCertifiedOptimization {ctx : ThreadCtx} (proof : KernelProof)
    (cert : CertifiedOptimization ctx) : KernelProof :=
  proof.addOptimizationMetadata cert.optimization

private def needsRaceProof (a b : AccessRange) : Bool :=
  a.base == b.base && (a.isWrite || b.isWrite) && !(a.isAtomic || b.isAtomic)

private def raceObligationsFor (access : AccessRange)
    (previous : Array AccessRange) : Array ProofObligation := Id.run do
  let mut out := #[]
  for prior in previous do
    if needsRaceProof prior access then
      out := out.push {
        kind := .disjoint prior access
        site := .accessPair prior.kind access.kind
      }
  return out

def addAccess (proof : KernelProof) (access : AccessRange) : KernelProof :=
  let newObligations := raceObligationsFor access proof.accesses
  { proof with
    accesses := proof.accesses.push access
    obligations := proof.obligations ++ newObligations
  }

def currentGuard (proof : KernelProof) : ThreadPred :=
  ThreadPred.andMany proof.guardStack

def shapeSyncCtx (proof : KernelProof) : Tyr.ShapeSync.ThreadCtx :=
  ShapeSyncBridge.threadCtx proof.threadCtx

def shapeSyncGraph (proof : KernelProof) : Tyr.ShapeSync.AccessGraph :=
  { nodes := proof.shapeSyncNodes.toList }

private def isDerivedShapeSyncAnalysisObligation : Tyr.ShapeSync.Obligation → Bool
  | .producerConsumer (.analysis _) => true
  | _ => false

def refreshShapeSyncAnalysis (proof : KernelProof) : KernelProof :=
  let preserved :=
    proof.shapeSyncObligations.filter fun obl =>
      !isDerivedShapeSyncAnalysisObligation obl
  if proof.shapeSyncNodes.isEmpty then
    { proof with
      shapeSyncGraph? := none
      shapeSyncProducerConsumer? := none
      shapeSyncObligations := preserved
    }
  else
    let graph := proof.shapeSyncGraph
    let analysis := Tyr.ShapeSync.ProducerConsumerAnalysis.build proof.shapeSyncCtx graph
    { proof with
      shapeSyncGraph? := some graph
      shapeSyncProducerConsumer? := some analysis
      shapeSyncObligations :=
        preserved.push (.producerConsumer (.analysis analysis))
    }

def addShapeSyncObligation (proof : KernelProof)
    (obl : Tyr.ShapeSync.Obligation) : KernelProof :=
  { proof with shapeSyncObligations := proof.shapeSyncObligations.push obl }

def addShapeSyncPipelineAnalysis (proof : KernelProof)
    (analysis : Tyr.ShapeSync.ProducerConsumerAnalysis) : KernelProof :=
  { proof with
    shapeSyncPipelineAnalyses := proof.shapeSyncPipelineAnalyses.push analysis
  }

private def addShapeSyncNode (proof : KernelProof)
    (role : Tyr.ShapeSync.NodeRole) (accesses : List Tyr.ShapeSync.Access) :
    KernelProof :=
  match ShapeSyncBridge.threadPred? proof.currentGuard with
  | none => proof
  | some guard =>
      { proof with
        shapeSyncNodes := proof.shapeSyncNodes.push {
          id := proof.shapeSyncNodes.size
          role
          guard
          accesses
        }
      }

private def addShapeSyncProducerWrite (proof : KernelProof)
    (kind : Tyr.ShapeSync.ProducerKind) (dst : VarId) : KernelProof :=
  proof.addShapeSyncNode (.producer kind) [Tyr.ShapeSync.Access.write dst.idx]

private def addShapeSyncConsumerReads (proof : KernelProof)
    (resources : List VarId) : KernelProof :=
  match resources with
  | [] => proof
  | _ =>
      proof.addShapeSyncNode .consumer
        (resources.map fun resource => Tyr.ShapeSync.Access.read resource.idx)

def pushGuard (proof : KernelProof) (guard : ThreadPred) : KernelProof :=
  { proof with guardStack := proof.guardStack.push guard }

def popGuard (proof : KernelProof) : KernelProof :=
  if proof.guardStack.isEmpty then proof
  else { proof with guardStack := proof.guardStack.pop }

def withThreadCtx (proof : KernelProof) (ctx : ThreadCtx) : KernelProof :=
  { proof with threadCtx := ctx }

private def addParticipantObligationFromStmt
    (proof : KernelProof) (kind : BarrierSiteKind) (barrierId numThreads : Nat) :
    KernelProof :=
  proof.addObligation {
    kind := .participantCount proof.currentGuard numThreads
    site := .barrier kind barrierId
  }

mutual

partial def completeFromStmt (proof : KernelProof) : KStmt → KernelProof
  | .declRT v dtype rows cols layout =>
      proof.insertVarSemantics v (.registerTile dtype rows cols layout)
  | .declST v dtype rows cols layout =>
      proof.insertVarSemantics v (.sharedTile dtype rows cols layout)
  | .declRV v dtype len =>
      proof.insertVarSemantics v (.registerVector dtype len)
  | .declSV v dtype len =>
      proof.insertVarSemantics v (.sharedVector dtype len)
  | .declSemaphore v =>
      proof.insertVarSemantics v .semaphore
  | .declTT v dtype rows cols =>
      proof.insertVarSemantics v (.tensorTile dtype rows cols)
  | .declTMEMPool v slots clusterSize managed =>
      proof.insertVarSemantics v (.tensorMemoryPool slots clusterSize managed)
  | .declGPtr v dtype _ =>
      proof.insertVarSemantics v (.globalPointer dtype)
  | .declKVal v _ _ =>
      proof.insertVarSemantics v .scalar
  | .constInt dst value =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst (.const value)
  | .getThreadIdx dst axis =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst (.threadIdx axis)
  | .getBlockIdx dst axis =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst (.blockIdx axis)
  | .getGridDim dst axis =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst (.gridDim axis)
  | .warpGroupIdx dst =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst .warpGroupIdx
  | .layoutDim dst src axis =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst (.layoutDim src axis)
  | .scalarUnary op dst src =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst (.unary op (proof.scalarOrVar src))
  | .scalarBinary op dst lhs rhs =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst (.binary op (proof.scalarOrVar lhs) (proof.scalarOrVar rhs))
  | .scalarCompare op dst lhs rhs =>
      (proof.insertVarSemantics dst .scalar).insertPredicate dst (.cmp op (proof.scalarOrVar lhs) (proof.scalarOrVar rhs))
  | .scalarSelect dst cond ifTrue ifFalse =>
      (proof.insertVarSemantics dst .scalar).insertScalar dst (.select
        (proof.predicateOrVar cond)
        (proof.scalarOrVar ifTrue)
        (proof.scalarOrVar ifFalse))
  | .load _dst src =>
      proof.addShapeSyncConsumerReads [src]
  | .loadAsync dst _src =>
      proof.addShapeSyncProducerWrite .cpAsync dst
  | .tmaLoad dst _src _coord =>
      proof.addShapeSyncProducerWrite .tma dst
  | .loadGlobalAsync dst _src _coordB _coordD _coordR _coordC _sem =>
      proof.addShapeSyncProducerWrite .cpAsync dst
  | .cpAsyncLoad dst _src _coordB _coordD _coordR _coordC _sem =>
      proof.addShapeSyncProducerWrite .cpAsync dst
  | .tmaLoadAsync dst _src _coordB _coordD _coordR _coordC _sem =>
      proof.addShapeSyncProducerWrite .tma dst
  | .clusterTmaLoad dst _src _coordB _coordD _coordR _coordC _sem =>
      proof.addShapeSyncProducerWrite .tma dst
  | .mma _trans _dst a b c =>
      proof.addShapeSyncConsumerReads [a, b, c]
  | .mm _trans _dst a b =>
      proof.addShapeSyncConsumerReads [a, b]
  | .tcgen05Mm _trans _dst a b =>
      proof.addShapeSyncConsumerReads [a, b]
  | .tcgen05Mma _trans _dst a b c =>
      proof.addShapeSyncConsumerReads [a, b, c]
  | .tcgen05MmaScaled _trans _dst a b c scaleA scaleB =>
      proof.addShapeSyncConsumerReads [a, b, c, scaleA, scaleB]
  | .loadScaleTmem _dst src _stage =>
      proof.addShapeSyncConsumerReads [src]
  | .tmemSubtile _dst src _offset =>
      proof.addShapeSyncConsumerReads [src]
  | .loadScalarGlobal dst src offset =>
      (proof.insertVarSemantics dst .scalar).addAccess {
        base := src
        offset := proof.scalarOrVar offset
        width := 1
        guard := proof.currentGuard
        isWrite := false
        kind := .loadScalarGlobal
      }
  | .storeScalarGlobal dst _src offset =>
      proof.addAccess {
        base := dst
        offset := proof.scalarOrVar offset
        width := 1
        guard := proof.currentGuard
        isWrite := true
        kind := .storeScalarGlobal
      }
  | .loadVecGlobal dst src offset =>
      proof.addAccess {
        base := src
        offset := proof.scalarOrVar offset
        width := proof.varElementCountD dst
        guard := proof.currentGuard
        isWrite := false
        kind := .loadVecGlobal
      }
  | .storeVecGlobal dst src offset =>
      proof.addAccess {
        base := dst
        offset := proof.scalarOrVar offset
        width := proof.varElementCountD src
        guard := proof.currentGuard
        isWrite := true
        kind := .storeVecGlobal
      }
  | .storeVecGlobalAdd dst src offset =>
      proof.addAccess {
        base := dst
        offset := proof.scalarOrVar offset
        width := proof.varElementCountD src
        guard := proof.currentGuard
        isWrite := true
        isAtomic := true
        kind := .storeVecGlobalAdd
      }
  | .namedBarrierSync id numThreads =>
      proof.addParticipantObligationFromStmt .namedBarrierSync id numThreads
  | .namedBarrierArrive id numThreads =>
      proof.addParticipantObligationFromStmt .namedBarrierArrive id numThreads
  | .ifStmt cond thenBody elseBody =>
      let pred := proof.predicateOrVar cond
      let afterThen := (proof.pushGuard pred).completeFromStmts thenBody |>.popGuard
      (afterThen.pushGuard (.not pred)).completeFromStmts elseBody |>.popGuard
  | .ifWarpGroup wgIdx body =>
      (proof.pushGuard (.warpGroupEq wgIdx)).completeFromStmts body |>.popGuard
  | .forLoop _ _ _ body
  | .forLoopVal _ _ _ body
  | .forLoopStride _ _ _ _ body
  | .forLoopRev _ _ _ body
  | .forLoopValRev _ _ _ body
  | .whileLoop body =>
      proof.completeFromStmts body
  | _ => proof

partial def completeFromStmts (proof : KernelProof) (stmts : Array KStmt) :
    KernelProof :=
  let completed := stmts.foldl (fun p stmt => p.completeFromStmt stmt) proof
  completed.refreshShapeSyncAnalysis

end

def shapeSyncAnalysisFromStmts (ctx : ThreadCtx) (stmts : Array KStmt) :
    Option Tyr.ShapeSync.ProducerConsumerAnalysis :=
  let proof := ({ (default : KernelProof) with threadCtx := ctx }).completeFromStmts stmts
  proof.shapeSyncProducerConsumer?

def diagnoseObligation (kernelName : String) (proof : KernelProof)
    (obl : ProofObligation) : Option ProofFailure :=
  match ProofDiagnostics.check? proof.threadCtx obl.kind with
  | some true => none
  | some false =>
      some {
        kernelName := kernelName
        site := obl.site
        guard := obl.kind.guard
        proposition := obl.kind
        reason := .false
      }
  | none =>
      some {
        kernelName := kernelName
        site := obl.site
        guard := obl.kind.guard
        proposition := obl.kind
        reason := .unsupported
      }

def diagnose (kernelName : String) (proof : KernelProof) : Array ProofFailure := Id.run do
  let mut out := #[]
  for obl in proof.obligations do
    match proof.diagnoseObligation kernelName obl with
    | none => pure ()
    | some failure => out := out.push failure
  return out

def failures (kernelName : String) (proof : KernelProof) : Array ProofFailure :=
  proof.diagnose kernelName

end KernelProof

end Tyr.GPU.Codegen
