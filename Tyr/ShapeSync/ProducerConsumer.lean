import Tyr.ShapeSync.Graph

/-!
# Generic producer/consumer barrier analysis

This module ports the reusable structure of TileLang-style producer/consumer
analysis:

- detect resource dependences from producer writes to consumer reads/accesses,
- compute wait/release windows in the consumer stream,
- pull waits forward when SIMT/cp.async producers write resources read earlier,
- detect mergeable pure-TMA barrier groups,
- derive participant-count obligations for forward and back-pressure arrivals.
-/

namespace Tyr.ShapeSync

structure BarrierProtocol where
  dependence : DependencyWindow
  barrierId : Nat
  forward : SyncObligation
  backpressure : SyncObligation
  waitPos : Nat
  releasePos : Nat
deriving Repr, BEq, DecidableEq

namespace BarrierProtocol

def Valid (ctx : ThreadCtx) (protocol : BarrierProtocol) : Prop :=
  protocol.forward.Valid ctx ∧
    protocol.backpressure.Valid ctx ∧
    protocol.forward.kind = .mbarrierArrive ∧
    protocol.backpressure.kind = .mbarrierArrive ∧
    protocol.forward.barrierId? = some protocol.barrierId ∧
    protocol.backpressure.barrierId? = some protocol.barrierId ∧
    protocol.waitPos = protocol.dependence.waitPos ∧
    protocol.releasePos = protocol.dependence.releasePos ∧
    protocol.waitPos <= protocol.releasePos

instance (ctx : ThreadCtx) (protocol : BarrierProtocol) :
    Decidable (protocol.Valid ctx) := by
  unfold Valid
  infer_instance

def fromWindow (ctx : ThreadCtx) (barrierId : Nat)
    (window : DependencyWindow) : BarrierProtocol :=
  let consumerGuard := window.consumer?.map (fun consumer => consumer.guard) |>.getD .top
  {
    dependence := window
    barrierId
    forward := {
      kind := .mbarrierArrive
      guard := window.producer.guard
      expectedParticipants := ctx.participantCount window.producer.guard
      barrierId? := some barrierId
    }
    backpressure := {
      kind := .mbarrierArrive
      guard := consumerGuard
      expectedParticipants := ctx.participantCount consumerGuard
      barrierId? := some barrierId
    }
    waitPos := window.waitPos
    releasePos := window.releasePos
  }

def render (protocol : BarrierProtocol) : String :=
  s!"barrier#{protocol.barrierId}: wait@{protocol.waitPos}, release@{protocol.releasePos}, " ++
  s!"resource={protocol.dependence.resource}"

end BarrierProtocol

def ProtocolsValid (ctx : ThreadCtx) : List BarrierProtocol → Prop
  | [] => True
  | protocol :: rest => protocol.Valid ctx ∧ ProtocolsValid ctx rest

def decidableProtocolsValid (ctx : ThreadCtx) :
    (protocols : List BarrierProtocol) → Decidable (ProtocolsValid ctx protocols)
  | [] => isTrue True.intro
  | protocol :: rest =>
      match inferInstanceAs (Decidable (protocol.Valid ctx)),
          decidableProtocolsValid ctx rest with
      | isTrue hp, isTrue hs => isTrue ⟨hp, hs⟩
      | isFalse hp, _ => isFalse (fun h => hp h.left)
      | _, isFalse hs => isFalse (fun h => hs h.right)

instance (ctx : ThreadCtx) (protocols : List BarrierProtocol) :
    Decidable (ProtocolsValid ctx protocols) :=
  decidableProtocolsValid ctx protocols

structure ProducerConsumerAnalysis where
  graph : AccessGraph
  windows : List DependencyWindow
  adjustedWindows : List DependencyWindow
  earliestAsyncRead : Nat
  canMergeBarriers : Bool
  protocols : List BarrierProtocol
deriving Repr, BEq, DecidableEq

namespace ProducerConsumerAnalysis

private def enumerateFrom (i : Nat) : List α → List (Nat × α)
  | [] => []
  | x :: xs => (i, x) :: enumerateFrom (i + 1) xs

def build (ctx : ThreadCtx) (graph : AccessGraph)
    (barrierBase : Nat := 0) : ProducerConsumerAnalysis :=
  let windows := graph.dependencyWindows
  let adjusted := graph.adjustedWindowsForAsync
  let protocols :=
    (enumerateFrom barrierBase adjusted).map fun (barrierId, window) =>
      BarrierProtocol.fromWindow ctx barrierId window
  {
    graph
    windows
    adjustedWindows := adjusted
    earliestAsyncRead := graph.earliestAsyncRead
    canMergeBarriers :=
      AccessGraph.canMergePureTmaBarriers adjusted graph.hasAsyncProducer
    protocols
  }

def WellFormed (analysis : ProducerConsumerAnalysis) : Prop :=
  analysis.windows = analysis.graph.dependencyWindows ∧
    analysis.adjustedWindows = analysis.graph.adjustedWindowsForAsync ∧
    analysis.earliestAsyncRead = analysis.graph.earliestAsyncRead ∧
    analysis.canMergeBarriers =
      AccessGraph.canMergePureTmaBarriers analysis.adjustedWindows
        analysis.graph.hasAsyncProducer ∧
    analysis.protocols.length = analysis.adjustedWindows.length

instance (analysis : ProducerConsumerAnalysis) : Decidable analysis.WellFormed := by
  unfold WellFormed
  infer_instance

def Valid (ctx : ThreadCtx) (analysis : ProducerConsumerAnalysis) : Prop :=
  analysis.WellFormed ∧ ProtocolsValid ctx analysis.protocols

instance (ctx : ThreadCtx) (analysis : ProducerConsumerAnalysis) :
    Decidable (analysis.Valid ctx) := by
  unfold Valid
  infer_instance

end ProducerConsumerAnalysis

inductive ProducerConsumerObligation where
| protocols (protocols : List BarrierProtocol)
| analysis (analysis : ProducerConsumerAnalysis)
deriving Repr, BEq, DecidableEq

namespace ProducerConsumerObligation

def Valid (ctx : ThreadCtx) : ProducerConsumerObligation → Prop
  | .protocols ps => ProtocolsValid ctx ps
  | .analysis a => a.Valid ctx

def decidableValid (ctx : ThreadCtx) :
    (obl : ProducerConsumerObligation) → Decidable (obl.Valid ctx)
  | .protocols ps => by
      unfold Valid
      infer_instance
  | .analysis a => by
      unfold Valid ProducerConsumerAnalysis.Valid
      infer_instance

instance (ctx : ThreadCtx) (obl : ProducerConsumerObligation) :
    Decidable (obl.Valid ctx) :=
  decidableValid ctx obl

end ProducerConsumerObligation

end Tyr.ShapeSync
