import Lean
import Tyr.ShapeSync.ProducerConsumer

/-!
# Shape/synchronization proof surface

`Obligation` is the generic proof object that lowering code can attach to a
decision.  `shape_sync` is the intended low-friction proof step: it tries the
native arithmetic/propositional automation Lean ships with before falling back
to definitional simplification of the closed finite participant-count checks.
-/

namespace Tyr.ShapeSync

inductive Obligation where
| shape (obl : ShapeObligation)
| sync (obl : SyncObligation)
| producerConsumer (obl : ProducerConsumerObligation)
| both (lhs rhs : Obligation)
deriving Repr, BEq, DecidableEq

namespace Obligation

def Valid (ctx : ThreadCtx) : Obligation → Prop
  | .shape obl => obl.Valid
  | .sync obl => obl.Valid ctx
  | .producerConsumer obl => obl.Valid ctx
  | .both lhs rhs => lhs.Valid ctx ∧ rhs.Valid ctx

def decidableValid (ctx : ThreadCtx) : (obl : Obligation) → Decidable (obl.Valid ctx)
  | .shape shapeObl => by
      unfold Valid
      infer_instance
  | .sync syncObl => by
      unfold Valid
      infer_instance
  | .producerConsumer pcObl => by
      unfold Valid
      infer_instance
  | .both lhs rhs =>
      match decidableValid ctx lhs, decidableValid ctx rhs with
      | isTrue hl, isTrue hr => isTrue ⟨hl, hr⟩
      | isFalse hl, _ => isFalse (fun h => hl h.left)
      | _, isFalse hr => isFalse (fun h => hr h.right)

instance (ctx : ThreadCtx) (obl : Obligation) : Decidable (obl.Valid ctx) :=
  decidableValid ctx obl

def render : Obligation → String
  | .shape (.rangeWithinShape range dim) =>
      s!"range [{range.min}, {range.min + range.extent}) within shape {dim}"
  | .shape (.localIndexInBounds range dim index) =>
      s!"index {range.min}+{index} in shape {dim}"
  | .shape (.nonUnitExtentsFit base src) =>
      s!"non-unit extents fit: base={repr base}, src={repr src}"
  | .shape (.divisible n d) =>
      s!"{n} divisible by {d}"
  | .sync obl => obl.render
  | .producerConsumer (.protocols protocols) =>
      s!"producer/consumer protocols: {protocols.map BarrierProtocol.render}"
  | .producerConsumer (.analysis analysis) =>
      s!"producer/consumer analysis: {analysis.protocols.map BarrierProtocol.render}"
  | .both lhs rhs => s!"{lhs.render}; {rhs.render}"

end Obligation

structure Certificate (ctx : ThreadCtx) (obl : Obligation) where
  proof : obl.Valid ctx

namespace Certificate

def ofProof (ctx : ThreadCtx) (obl : Obligation) (proof : obl.Valid ctx) :
    Certificate ctx obl :=
  { proof }

end Certificate

syntax (name := shapeSyncTac) "shape_sync" : tactic

macro_rules
  | `(tactic| shape_sync) =>
      `(tactic|
        solve
        | omega
        | grind
        | native_decide
        | simp [Obligation.Valid, ShapeObligation.Valid, SyncObligation.Valid,
            ProducerConsumerObligation.Valid, ProducerConsumerAnalysis.Valid,
            ProducerConsumerAnalysis.WellFormed, ProtocolsValid, BarrierProtocol.Valid,
            ParticipantCount, ThreadCtx.participantCount, ThreadCtx.threads,
            ThreadPred.eval, ThreadId.axis, ThreadId.linear, ThreadId.warpGroup,
            DimRange.WithinShape, DimRange.ContainsLocal, DimRange.globalIndex,
            NonUnitExtentsFit, nonUnitExtentsFit, DivisibleBy, MultipleOf] <;> omega)

end Tyr.ShapeSync
