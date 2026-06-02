import Tyr.ShapeSync

namespace Tests.ShapeSync.TestCore

open Tyr.ShapeSync

private def cta256 : ThreadCtx :=
  { blockDimX := 256, blockDimY := 1, blockDimZ := 1, warpSize := 32, warpGroupSize := 128 }

#guard ProofBackend.smtFacade.render == "smt_facade"
#guard ProofStrategy.default.uses .omega
#guard ProofStrategy.default.uses .smtFacade

private theorem symbolicRangeAccessInBounds
    (base extent shape i : Int)
    (hbase : 0 <= base)
    (hupper : base + extent <= shape)
    (hi0 : 0 <= i)
    (hie : i < extent) :
    0 <= base + i ∧ base + i < shape := by
  shape_sync

private theorem symbolicRangeAccessInBoundsNative
    (base extent shape i : Int)
    (hbase : 0 <= base)
    (hupper : base + extent <= shape)
    (hi0 : 0 <= i)
    (hie : i < extent) :
    0 <= base + i ∧ base + i < shape := by
  shape_sync_native

private theorem symbolicRangeAccessInBoundsSmtFacade
    (base extent shape i : Int)
    (hbase : 0 <= base)
    (hupper : base + extent <= shape)
    (hi0 : 0 <= i)
    (hie : i < extent) :
    0 <= base + i ∧ base + i < shape := by
  shape_sync_smt

private theorem dimRangeAccessInBounds
    (range : DimRange) (shape i : Int)
    (hr : range.WithinShape shape)
    (hi : range.ContainsLocal i) :
    0 <= range.globalIndex i ∧ range.globalIndex i < shape :=
  DimRange.local_index_in_shape hr hi

#guard nonUnitExtentsFit [1, 64, 1, 16] [64, 1, 16] == true
#guard nonUnitExtentsFit [1, 32, 1, 16] [64, 1, 16] == false

private theorem nonUnitShapeCompatibility :
    (ShapeObligation.nonUnitExtentsFit [1, 64, 1, 16] [64, 1, 16]).Valid := by
  shape_sync

#guard cta256.participantCount (.warpGroupEq 0) == 128
#guard cta256.participantCount (.warpGroupEq 1) == 128
#guard cta256.participantCount (.modEq 0 4 0) == 64

private theorem warpGroupNamedBarrierValid :
    (SyncObligation.namedSync 2 128 (.warpGroupEq 0)).Valid cta256 := by
  shape_sync

private theorem warpGroupNamedBarrierValidFinite :
    (SyncObligation.namedSync 2 128 (.warpGroupEq 0)).Valid cta256 := by
  shape_sync_finite

private theorem moduloLaneParticipantCount :
    ParticipantCount cta256 (.modEq 0 4 0) 64 := by
  shape_sync

private def guardedBarrier : Obligation :=
  .sync (SyncObligation.namedArrive 3 64 (.modEq 0 4 0))

private theorem guardedBarrierCertificate :
    guardedBarrier.Valid cta256 := by
  shape_sync

private def shapeAndSync : Obligation :=
  .both
    (.shape (.nonUnitExtentsFit [1, 64, 1, 16] [64, 1, 16]))
    (.sync (SyncObligation.namedSync 4 128 (.warpGroupEq 1)))

private theorem combinedCertificate :
    shapeAndSync.Valid cta256 := by
  shape_sync

private def tmaLeader : ThreadPred :=
  .and (.warpGroupEq 0) (.axisEq 0 0)

private def consumerGroup : ThreadPred :=
  .warpGroupEq 1

private def tmaProducer : GraphNode :=
  { id := 0, role := .producer .tma, guard := tmaLeader, accesses := [Access.write 0] }

private def consumer : GraphNode :=
  { id := 1, role := .consumer, guard := consumerGroup, accesses := [Access.read 0] }

private def pcGraph : AccessGraph :=
  { nodes := [tmaProducer, consumer] }

private def pcAnalysis : ProducerConsumerAnalysis :=
  ProducerConsumerAnalysis.build cta256 pcGraph

private def headForwardParticipants (analysis : ProducerConsumerAnalysis) : Option Nat :=
  match analysis.protocols.head? with
  | some protocol => some protocol.forward.expectedParticipants
  | none => none

private def headBackpressureParticipants (analysis : ProducerConsumerAnalysis) : Option Nat :=
  match analysis.protocols.head? with
  | some protocol => some protocol.backpressure.expectedParticipants
  | none => none

#guard pcGraph.dependencyWindows.length == 1
#guard headForwardParticipants pcAnalysis == some 1
#guard headBackpressureParticipants pcAnalysis == some 128

private theorem producerConsumerAnalysisValid :
    (.producerConsumer (.analysis pcAnalysis) : Obligation).Valid cta256 := by
  shape_sync

private def asyncProducer : GraphNode :=
  { id := 2, role := .producer .cpAsync, guard := .warpGroupEq 0,
    accesses := [Access.write 1] }

private def asyncConsumer : GraphNode :=
  { id := 3, role := .consumer, guard := consumerGroup, accesses := [Access.read 1] }

private def lateTmaConsumer : GraphNode :=
  { id := 4, role := .consumer, guard := consumerGroup, accesses := [Access.read 0] }

private def asyncGraph : AccessGraph :=
  { nodes := [tmaProducer, asyncProducer, asyncConsumer, lateTmaConsumer] }

private def waitPositions (windows : List DependencyWindow) : List Nat :=
  windows.map fun window => window.waitPos

#guard waitPositions asyncGraph.dependencyWindows == [1, 0]
#guard waitPositions asyncGraph.adjustedWindowsForAsync == [0, 0]

private def mergeTmaA : GraphNode :=
  { id := 5, role := .producer .tma, guard := tmaLeader, accesses := [Access.write 2] }

private def mergeTmaB : GraphNode :=
  { id := 6, role := .producer .tma, guard := tmaLeader, accesses := [Access.write 3] }

private def mergeConsumer : GraphNode :=
  { id := 7, role := .consumer, guard := consumerGroup,
    accesses := [Access.read 2, Access.read 3] }

private def mergeGraph : AccessGraph :=
  { nodes := [mergeTmaA, mergeTmaB, mergeConsumer] }

private def mergeAnalysis : ProducerConsumerAnalysis :=
  ProducerConsumerAnalysis.build cta256 mergeGraph

#guard mergeAnalysis.canMergeBarriers == true

private theorem mergeAnalysisValid :
    (.producerConsumer (.analysis mergeAnalysis) : Obligation).Valid cta256 := by
  shape_sync

end Tests.ShapeSync.TestCore
