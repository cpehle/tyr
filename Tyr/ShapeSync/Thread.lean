import Tyr.ShapeSync.Shape

/-!
# Generic thread-domain and synchronization obligations

This module models the finite participant sets that synchronization lowering
cares about.  It is deliberately backend-neutral: CUDA named barriers,
warp-group specialization, and block-wide vote/barrier intrinsics can all be
described as obligations over `ThreadCtx` and `ThreadPred`.
-/

namespace Tyr.ShapeSync

structure ThreadCtx where
  blockDimX : Nat := 256
  blockDimY : Nat := 1
  blockDimZ : Nat := 1
  warpSize : Nat := 32
  warpGroupSize : Nat := 128
deriving Repr, BEq, DecidableEq

namespace ThreadCtx

def totalThreads (ctx : ThreadCtx) : Nat :=
  ctx.blockDimX * ctx.blockDimY * ctx.blockDimZ

def axisBound (ctx : ThreadCtx) : Nat → Nat
  | 0 => ctx.blockDimX
  | 1 => ctx.blockDimY
  | _ => ctx.blockDimZ

end ThreadCtx

structure ThreadId where
  x : Nat
  y : Nat := 0
  z : Nat := 0
deriving Repr, BEq, DecidableEq

namespace ThreadId

def axis (tid : ThreadId) : Nat → Nat
  | 0 => tid.x
  | 1 => tid.y
  | _ => tid.z

def inBounds (ctx : ThreadCtx) (tid : ThreadId) : Bool :=
  tid.x < ctx.blockDimX && tid.y < ctx.blockDimY && tid.z < ctx.blockDimZ

def linear (ctx : ThreadCtx) (tid : ThreadId) : Nat :=
  tid.x + ctx.blockDimX * (tid.y + ctx.blockDimY * tid.z)

def warpGroup (ctx : ThreadCtx) (tid : ThreadId) : Nat :=
  tid.linear ctx / ctx.warpGroupSize

end ThreadId

inductive ThreadPred where
| top
| bottom
| axisEq (axis value : Nat)
| axisLt (axis upper : Nat)
| linearLt (upper : Nat)
| linearGe (lower : Nat)
| warpGroupEq (idx : Nat)
| modEq (axis modulus residue : Nat)
| and (lhs rhs : ThreadPred)
| or (lhs rhs : ThreadPred)
| not (arg : ThreadPred)
deriving Repr, BEq, DecidableEq

namespace ThreadPred

def warpGroup (idx : Nat) : ThreadPred :=
  .warpGroupEq idx

def eval : ThreadPred → ThreadCtx → ThreadId → Bool
  | .top, _, _ => true
  | .bottom, _, _ => false
  | .axisEq axis value, _, tid => tid.axis axis == value
  | .axisLt axis upper, _, tid => tid.axis axis < upper
  | .linearLt upper, ctx, tid => tid.linear ctx < upper
  | .linearGe lower, ctx, tid => lower <= tid.linear ctx
  | .warpGroupEq idx, ctx, tid => tid.warpGroup ctx == idx
  | .modEq axis modulus residue, _, tid =>
      modulus != 0 && tid.axis axis % modulus == residue % modulus
  | .and lhs rhs, ctx, tid => lhs.eval ctx tid && rhs.eval ctx tid
  | .or lhs rhs, ctx, tid => lhs.eval ctx tid || rhs.eval ctx tid
  | .not arg, ctx, tid => !(arg.eval ctx tid)

def render : ThreadPred → String
  | .top => "true"
  | .bottom => "false"
  | .axisEq 0 value => s!"threadIdx.x == {value}"
  | .axisEq 1 value => s!"threadIdx.y == {value}"
  | .axisEq _ value => s!"threadIdx.z == {value}"
  | .axisLt 0 upper => s!"threadIdx.x < {upper}"
  | .axisLt 1 upper => s!"threadIdx.y < {upper}"
  | .axisLt _ upper => s!"threadIdx.z < {upper}"
  | .linearLt upper => s!"linearThreadId < {upper}"
  | .linearGe lower => s!"{lower} <= linearThreadId"
  | .warpGroupEq idx => s!"warpGroup == {idx}"
  | .modEq 0 modulus residue => s!"threadIdx.x % {modulus} == {residue}"
  | .modEq 1 modulus residue => s!"threadIdx.y % {modulus} == {residue}"
  | .modEq _ modulus residue => s!"threadIdx.z % {modulus} == {residue}"
  | .and lhs rhs => s!"({lhs.render} && {rhs.render})"
  | .or lhs rhs => s!"({lhs.render} || {rhs.render})"
  | .not arg => s!"!({arg.render})"

end ThreadPred

namespace ThreadCtx

private def concatMap (xs : List α) (f : α → List β) : List β :=
  xs.foldr (fun x acc => f x ++ acc) []

def threads (ctx : ThreadCtx) : List ThreadId :=
  concatMap (List.range ctx.blockDimZ) fun z =>
    concatMap (List.range ctx.blockDimY) fun y =>
      (List.range ctx.blockDimX).map fun x => { x, y, z }

def participantCount (ctx : ThreadCtx) (pred : ThreadPred) : Nat :=
  (ctx.threads.filter fun tid => pred.eval ctx tid).length

end ThreadCtx

def ParticipantCount (ctx : ThreadCtx) (pred : ThreadPred) (expected : Nat) : Prop :=
  ctx.participantCount pred = expected

instance (ctx : ThreadCtx) (pred : ThreadPred) (expected : Nat) :
    Decidable (ParticipantCount ctx pred expected) := by
  unfold ParticipantCount
  infer_instance

inductive SyncKind where
| blockSync
| warpSync
| namedBarrierSync
| namedBarrierArrive
| mbarrierArrive
| mbarrierWait
deriving Repr, BEq, DecidableEq

namespace SyncKind

def render : SyncKind → String
  | .blockSync => "block_sync"
  | .warpSync => "warp_sync"
  | .namedBarrierSync => "named_barrier_sync"
  | .namedBarrierArrive => "named_barrier_arrive"
  | .mbarrierArrive => "mbarrier_arrive"
  | .mbarrierWait => "mbarrier_wait"

end SyncKind

structure SyncObligation where
  kind : SyncKind
  guard : ThreadPred := .top
  expectedParticipants : Nat
  barrierId? : Option Nat := none
deriving Repr, BEq, DecidableEq

namespace SyncObligation

def namedSync (id expected : Nat) (guard : ThreadPred := .top) : SyncObligation :=
  { kind := .namedBarrierSync, guard, expectedParticipants := expected, barrierId? := some id }

def namedArrive (id expected : Nat) (guard : ThreadPred := .top) : SyncObligation :=
  { kind := .namedBarrierArrive, guard, expectedParticipants := expected, barrierId? := some id }

def blockSync (ctx : ThreadCtx) (guard : ThreadPred := .top) : SyncObligation :=
  { kind := .blockSync, guard, expectedParticipants := ctx.totalThreads }

def Valid (ctx : ThreadCtx) (obl : SyncObligation) : Prop :=
  ParticipantCount ctx obl.guard obl.expectedParticipants

instance (ctx : ThreadCtx) (obl : SyncObligation) : Decidable (obl.Valid ctx) := by
  unfold Valid
  infer_instance

def diagnostic (ctx : ThreadCtx) (obl : SyncObligation) : Bool :=
  ctx.participantCount obl.guard == obl.expectedParticipants

def render (obl : SyncObligation) : String :=
  let id :=
    match obl.barrierId? with
    | none => ""
    | some n => s!"#{n}"
  s!"{obl.kind.render}{id}: participant_count({obl.guard.render}) = {obl.expectedParticipants}"

end SyncObligation

end Tyr.ShapeSync
