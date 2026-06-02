import LeanTest
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.Ops
import Tyr.GPU.Codegen.GlobalLayout
import Tyr.GPU.Codegen.Pipeline
import Tyr.GPU.Codegen.Proof

namespace Tests.GPUProof

open LeanTest
open Tyr.GPU
open Tyr.GPU.Codegen

private def outParam : KParam :=
  { name := "out", dtype := .Float32, isPointer := true }

private def outPtr : GPtr .Float32 :=
  ⟨⟨0⟩, "out"⟩

private def txMod4Eq0 : ThreadPred :=
  .cmp .Eq
    (.binary .Mod (.threadIdx 0) (.const 4))
    (.const 0)

private theorem grindPredicateSelfImplicationHolds :
    (ProofObligationKind.implication txMod4Eq0 txMod4Eq0).Holds
      ({ blockDimX := 256 } : ThreadCtx) := by
  intro tx ty tz _ _ _ h
  grind

private theorem scalarAddZeroHolds :
    (ProofObligationKind.scalarEquality .top
      (ScalarExpr.add (.threadIdx 0) (.const 0)) (.threadIdx 0)).Holds
      ({ blockDimX := 256 } : ThreadCtx) := by
  intro tx ty tz _ _ _ _
  exists Int.ofNat tx

private theorem guardedPredicateSelfHolds :
    (ProofObligationKind.predicateValue txMod4Eq0 txMod4Eq0 true).Holds
      ({ blockDimX := 256 } : ThreadCtx) := by
  intro tx ty tz _ _ _ h
  exact h

private theorem constantAlignmentHolds :
    (ProofObligationKind.scalarDivisible .top (.const 8) 4).Holds
      ({ blockDimX := 256 } : ThreadCtx) := by
  constructor
  · decide
  · intro tx ty tz _ _ _ _
    exists 8

private theorem shapeSyncWarpGroup0NamedSyncHolds :
    (Tyr.ShapeSync.SyncObligation.namedSync 11 128 (.warpGroupEq 0)).Valid
      (ShapeSyncBridge.threadCtx ({ blockDimX := 256 } : ThreadCtx)) := by
  native_decide

private partial def collectSemaphoreOps (stmts : Array KStmt) : Array SemaphoreOp :=
  stmts.foldl (init := #[]) fun out stmt =>
    match stmt with
    | .semaphore op _ => out.push op
    | .forLoop _ _ _ body
    | .forLoopVal _ _ _ body
    | .forLoopStride _ _ _ _ body
    | .forLoopRev _ _ _ body
    | .forLoopValRev _ _ _ body
    | .whileLoop body
    | .ifWarpGroup _ body =>
        out ++ collectSemaphoreOps body
    | .ifStmt _ thenBody elseBody =>
        out ++ collectSemaphoreOps thenBody ++ collectSemaphoreOps elseBody
    | _ => out

@[test]
def testParticipantCountModuloGuard : IO Unit := do
  let ctx : ThreadCtx := { blockDimX := 256 }
  assertTrue (participantCount? ctx txMod4Eq0 == some 64)
    "threadIdx.x % 4 == 0 should select 64 threads from a 256-thread CTA"

@[test]
def testScalarPredicateTrackingFromBuilder : IO Unit := do
  let kernel := buildKernelM "proof_scalar_tracking" .SM90 #[] do
    let tx : KVal UInt32 := ⟨← getThreadIdxX, "tx"⟩
    let four ← constIntVal 4 "four"
    let zero ← constIntVal 0 "zero"
    let rem ← scalarMod tx four "rem"
    let pred ← scalarEq rem zero "pred"
    comment pred.id.toIdent

  match kernel.proof.predicate? ⟨4⟩ with
  | some pred =>
      assertTrue (participantCount? kernel.proof.threadCtx pred == some 64)
        "builder should preserve the structured modulo predicate"
  | none =>
      fail "expected scalarEq result to be recorded as a structured predicate"

@[test]
def testNamedBarrierParticipantCountAcceptsWarpGroup : IO Unit := do
  let kernel := buildKernelM "proof_named_barrier_ok" .SM90 #[] do
    ifWarpGroup 0 do
      namedBarrierSync 2 128

  let failures := kernel.proof.diagnose kernel.name
  assertTrue failures.isEmpty
    s!"expected named barrier diagnostics to pass, got {repr failures}"

@[test]
def testNamedBarrierParticipantCountRejectsWrongCount : IO Unit := do
  let kernel := buildKernelM "proof_named_barrier_bad" .SM90 #[] do
    ifWarpGroup 0 do
      namedBarrierSync 2 64

  let failures := kernel.proof.diagnose kernel.name
  assertTrue (!failures.isEmpty)
    "expected incorrect named barrier participant count to produce a diagnostic failure"

@[test]
def testProofFactsDerivedFromDirectKStmtTree : IO Unit := do
  let kernel := buildKernelM "proof_direct_kstmt_guard" .SM90 #[] do
    let wg ← freshVar
    emit (.warpGroupIdx wg)
    let zero ← freshVar
    emit (.constInt zero 0)
    let cond ← freshVar
    emit (.scalarCompare .Eq cond wg zero)
    emit (.ifStmt cond #[.namedBarrierSync 5 128] #[])

  assertTrue (kernel.proof.obligations.size == 1)
    "KStmt traversal should derive one named-barrier participant obligation"
  let failures := kernel.proof.diagnose kernel.name
  assertTrue failures.isEmpty
    s!"expected direct KStmt proof facts to diagnose cleanly, got {repr failures}"

@[test]
def testAccessDisjointnessQueuedAsStoresAreRecorded : IO Unit := do
  let kernel := buildKernelM "proof_access_disjoint" .SM90 #[outParam] do
    let sv ← allocSV .Float32 16
    let off0 ← constIntVal 0 "off0"
    let off32 ← constIntVal 32 "off32"
    storeVecGlobalCoord outPtr sv off0.id
    storeVecGlobalCoord outPtr sv off32.id

  assertTrue (kernel.proof.accesses.size == 2)
    "expected both vector stores to be recorded"
  assertTrue (kernel.proof.obligations.size == 1)
    "second same-base store should queue one disjointness obligation"
  let failures := kernel.proof.diagnose kernel.name
  assertTrue failures.isEmpty
    s!"expected disjoint-store diagnostics to pass, got {repr failures}"

@[test]
def testAccessFactsDerivedFromDirectKStmtTree : IO Unit := do
  let kernel := buildKernelM "proof_direct_kstmt_access" .SM90 #[outParam] do
    let sv ← freshVar
    emit (.declSV sv .Float32 16)
    let off0 ← freshVar
    emit (.constInt off0 0)
    let off32 ← freshVar
    emit (.constInt off32 32)
    emit (.storeVecGlobal outPtr.id sv off0)
    emit (.storeVecGlobal outPtr.id sv off32)

  assertTrue (kernel.proof.accesses.size == 2)
    "KStmt traversal should derive both vector-store access ranges"
  assertTrue (kernel.proof.obligations.size == 1)
    "direct same-base stores should queue one disjointness obligation"
  let failures := kernel.proof.diagnose kernel.name
  assertTrue failures.isEmpty
    s!"expected direct KStmt disjointness diagnostics to pass, got {repr failures}"

@[test]
def testOptimizationDecisionCarriesSemanticCertificate : IO Unit := do
  let ctx : ThreadCtx := { blockDimX := 256 }
  let cert := CertifiedOptimization.scalarRewrite ctx `simplify `add_zero
    .top (ScalarExpr.add (.threadIdx 0) (.const 0)) (.threadIdx 0)
    scalarAddZeroHolds
  let proof := (default : KernelProof).addCertifiedOptimization cert

  assertTrue (proof.optimizations.size == 1)
    "optimization metadata should be recorded only after a semantic certificate exists"
  assertTrue (cert.certificate.obligation.kind == proof.obligations[0]!.kind)
    "certificate should target the recorded optimization obligation"

@[test]
def testAccessDisjointnessRejectsAmbiguousRace : IO Unit := do
  let kernel := buildKernelM "proof_access_race" .SM90 #[outParam] do
    let sv ← allocSV .Float32 16
    let off0 ← constIntVal 0 "off0"
    storeVecGlobalCoord outPtr sv off0.id
    storeVecGlobalCoord outPtr sv off0.id

  let failures := kernel.proof.diagnose kernel.name
  assertTrue (!failures.isEmpty) "expected at least one race diagnostic failure"

@[test]
def testOptimizationScalarRewriteCertificate : IO Unit := do
  let before := ScalarExpr.add (.threadIdx 0) (.const 0)
  let after := .threadIdx 0
  let ctx : ThreadCtx := { blockDimX := 256 }
  let cert := CertifiedOptimization.scalarRewrite ctx `simplify `add_zero
    .top before after scalarAddZeroHolds
  let proof := (default : KernelProof).addCertifiedOptimization cert

  assertTrue (proof.optimizations.size == 1)
    "optimization proof metadata should keep the typed rewrite claim"
  assertTrue (cert.certificate.obligation.kind == proof.obligations[0]!.kind)
    "scalar rewrite certificate should target the queued obligation"

@[test]
def testOptimizationRejectsInvalidScalarRewrite : IO Unit := do
  let before := ScalarExpr.add (.threadIdx 0) (.const 1)
  let after := .threadIdx 0
  let opt := OptimizationProof.scalarRewrite `simplify `bad_add_one .top before after
  match ProofDiagnostics.check? ({ blockDimX := 256 } : ThreadCtx)
      (opt.claim.obligation opt.site).kind with
  | some false => pure ()
  | other => fail s!"expected invalid scalar rewrite diagnostics to reject it, got {repr other}"

@[test]
def testOptimizationPredicateSimplificationCertificate : IO Unit := do
  let ctx : ThreadCtx := { blockDimX := 256 }
  let cert := CertifiedOptimization.predicateSimplification ctx
    `simplify `guarded_branch_true txMod4Eq0 txMod4Eq0 true
    guardedPredicateSelfHolds
  let proof := (default : KernelProof).addCertifiedOptimization cert

  assertTrue (proof.optimizations.size == 1)
    "predicate simplification should be recorded after a semantic certificate exists"

@[test]
def testOptimizationAlignmentCertificateThroughKernelM : IO Unit := do
  let kernel := buildKernelM "proof_opt_alignment" .SM90 #[] do
    addOptimizationProof <|
      CertifiedOptimization.alignmentCheck ({ blockDimX := 256 } : ThreadCtx)
        `vectorize `constant_offset_aligned .top (.const 8) 4
        constantAlignmentHolds

  assertTrue (kernel.proof.optimizations.size == 1)
    "KernelM should record optimization proof metadata"

@[test]
def testSyncOmissionDecisionRequiresProof : IO Unit := do
  let ctx : ThreadCtx := { blockDimX := 256 }
  let cert := CertifiedOptimization.syncOmission ctx
    `pipeline `omit_redundant_sync txMod4Eq0 txMod4Eq0
    guardedPredicateSelfHolds
  let decision : OptimizationDecision ctx Unit :=
    .accepted () cert

  assertTrue decision.isAccepted
    "sync omission should be modeled as an accepted optimization only with a semantic proof"

@[test]
def testProofOracleProducesSemanticProofResult : IO Unit := do
  let ctx : ThreadCtx := { blockDimX := 256 }
  let result := ProofOracle.proveUnder ctx (.optimization { pass := `simplify, rule := `guard })
    txMod4Eq0 txMod4Eq0 guardedPredicateSelfHolds

  assertTrue result.isProved
    "proof oracle should return a proved result only from a semantic Holds certificate"

@[test]
def testShapeSyncBridgeConvertsClosedModuloGuard : IO Unit := do
  let ctx : ThreadCtx := { blockDimX := 256 }
  match ShapeSyncBridge.threadPred? txMod4Eq0 with
  | some pred =>
      let shapeCtx := ShapeSyncBridge.threadCtx ctx
      assertTrue (shapeCtx.participantCount pred == 64)
        "ShapeSync bridge should preserve closed modulo participant counts"
  | none =>
      fail "expected closed modulo thread predicate to convert to ShapeSync"

@[test]
def testShapeSyncBridgeRejectsRuntimeBoolGuard : IO Unit := do
  match ShapeSyncBridge.threadPred? (.boolVar ⟨0⟩) with
  | none => pure ()
  | some pred =>
      fail s!"runtime boolean guard should not convert to ShapeSync, got {pred.render}"

@[test]
def testShapeSyncAnalysisDerivedFromKStmtTree : IO Unit := do
  let kernel := buildKernelM "proof_shapesync_pc" .SM90 #[] do
    let shared ← freshVar
    emit (.declST shared .Float32 16 16 .Row)
    let reg ← freshVar
    emit (.declRT reg .Float32 16 16 .Row)
    let global ← freshVar
    emit (.declGPtr global .Float32 "global")
    let coord ← freshVar
    emit (.constInt coord 0)
    let sem ← freshVar
    emit (.declSemaphore sem)
    emit (.ifWarpGroup 0 #[
      .tmaLoadAsync shared global coord coord coord coord sem
    ])
    emit (.ifWarpGroup 1 #[
      .load reg shared
    ])

  assertTrue (kernel.proof.shapeSyncNodes.size == 2)
    "expected ShapeSync to collect one producer node and one consumer node"
  match kernel.proof.shapeSyncProducerConsumer? with
  | some analysis =>
      assertTrue (analysis.protocols.length == 1)
        "expected one producer/consumer protocol for the shared resource"
      match analysis.protocols.head? with
      | some protocol =>
          assertTrue (protocol.forward.expectedParticipants == 128)
            "producer warp-group should contribute 128 participants"
          assertTrue (protocol.backpressure.expectedParticipants == 128)
            "consumer warp-group should contribute 128 participants"
          match kernel.proof.shapeSyncObligations[0]? with
          | some obl =>
              assertTrue (decide (obl.Valid
                  (ShapeSyncBridge.threadCtx kernel.proof.threadCtx)))
                "attached ShapeSync producer/consumer obligation should be valid"
          | none =>
              fail "expected a ShapeSync obligation"
      | none =>
          fail "expected a ShapeSync barrier protocol"
  | none =>
      fail "expected ShapeSync producer/consumer analysis to be attached to KernelProof"

@[test]
def testShapeSyncCheckedNamedBarrierRecordsObligation : IO Unit := do
  let shapeCtx := ShapeSyncBridge.threadCtx ({ blockDimX := 256 } : ThreadCtx)
  let kernel := buildKernelM "proof_shapesync_checked_barrier" .SM90 #[] do
    ifWarpGroup 0 do
      namedBarrierSyncChecked shapeCtx 11 128 (.warpGroupEq 0)
        shapeSyncWarpGroup0NamedSyncHolds

  assertTrue (kernel.proof.shapeSyncObligations.size == 1)
    "checked named barrier should record one ShapeSync obligation"
  match kernel.proof.shapeSyncObligations[0]? with
  | some (Tyr.ShapeSync.Obligation.sync obl) =>
      assertTrue (obl.barrierId? == some 11)
        "recorded ShapeSync sync obligation should retain the barrier id"
      assertTrue (decide (obl.Valid shapeCtx))
        "recorded ShapeSync sync obligation should be valid"
  | some other =>
      fail s!"expected a sync ShapeSync obligation, got {other.render}"
  | none =>
      fail "expected a ShapeSync obligation"

@[test]
def testShapeSyncPipelinePreviewAnalysisRecorded : IO Unit := do
  let kernel := buildKernelM "proof_shapesync_pipeline_preview" .SM90 #[] do
    let shared ← freshVar
    emit (.declST shared .Float32 16 16 .Row)
    let reg ← freshVar
    emit (.declRT reg .Float32 16 16 .Row)
    let global ← freshVar
    emit (.declGPtr global .Float32 "global")
    let coord ← freshVar
    emit (.constInt coord 0)
    let sem ← freshVar
    emit (.declSemaphore sem)
    pipelinedRingLoop { numIters := 2, depth := 1, warpSpecialized := true }
      (fun _ => emit (.tmaLoadAsync shared global coord coord coord coord sem))
      (fun _ => emit (.load reg shared))

  assertTrue (kernel.proof.shapeSyncPipelineAnalyses.size == 1)
    "warp-specialized pipeline should record one representative ShapeSync analysis"
  match kernel.proof.shapeSyncPipelineAnalyses[0]? with
  | some analysis =>
      assertTrue (analysis.protocols.length == 1)
        "representative pipeline analysis should have one shared-resource protocol"
      match analysis.protocols.head? with
      | some protocol =>
          assertTrue (protocol.forward.expectedParticipants == 128)
            "pipeline producer warp-group should contribute 128 participants"
          assertTrue (protocol.backpressure.expectedParticipants == 128)
            "pipeline consumer warp-group should contribute 128 participants"
      | none =>
          fail "expected a representative pipeline protocol"
  | none =>
      fail "expected a representative ShapeSync pipeline analysis"

@[test]
def testShapeSyncPipelineDepthTwoPreviewAnalysesRecorded : IO Unit := do
  let kernel := buildKernelM "proof_shapesync_pipeline_depth_two_preview" .SM90 #[] do
    let shared0 ← freshVar
    emit (.declST shared0 .Float32 16 16 .Row)
    let shared1 ← freshVar
    emit (.declST shared1 .Float32 16 16 .Row)
    let reg ← freshVar
    emit (.declRT reg .Float32 16 16 .Row)
    let global ← freshVar
    emit (.declGPtr global .Float32 "global")
    let coord ← freshVar
    emit (.constInt coord 0)
    let sem ← freshVar
    emit (.declSemaphore sem)
    let sharedAtStage (stage : Nat) := if stage % 2 == 0 then shared0 else shared1
    pipelinedRingLoop { numIters := 4, depth := 2, warpSpecialized := true }
      (fun stage => emit (.tmaLoadAsync (sharedAtStage stage) global coord coord coord coord sem))
      (fun stage => emit (.load reg (sharedAtStage stage)))

  assertTrue (kernel.proof.shapeSyncPipelineAnalyses.size == 2)
    "depth-2 warp-specialized pipeline should record one ShapeSync analysis per stage"
  for analysis in kernel.proof.shapeSyncPipelineAnalyses do
    assertTrue (analysis.protocols.length == 1)
      "each depth-2 representative stage should have one producer/consumer protocol"

@[test]
def testShapeSyncPipelineDepthTwoHandoffFallsBackToConservativeSync : IO Unit := do
  let kernel := buildKernelM "proof_shapesync_pipeline_depth_two_handoff_fallback" .SM90 #[] do
    let shared0 ← freshVar
    emit (.declST shared0 .Float32 16 16 .Row)
    let shared1 ← freshVar
    emit (.declST shared1 .Float32 16 16 .Row)
    let reg ← freshVar
    emit (.declRT reg .Float32 16 16 .Row)
    let global ← freshVar
    emit (.declGPtr global .Float32 "global")
    let coord ← freshVar
    emit (.constInt coord 0)
    let sem ← freshVar
    emit (.declSemaphore sem)
    let sharedAtStage (stage : Nat) := if stage % 2 == 0 then shared0 else shared1
    pipelinedRingLoop
      { numIters := 4, depth := 2, warpSpecialized := true, useShapeSyncBarriers := true }
      (fun stage => emit (.tmaLoadAsync (sharedAtStage stage) global coord coord coord coord sem))
      (fun stage => emit (.load reg (sharedAtStage stage)))

  assertTrue (kernel.proof.shapeSyncPipelineAnalyses.size == 2)
    "depth-2 opt-in should still record per-stage ShapeSync analysis"
  assertTrue (collectSemaphoreOps kernel.body).isEmpty
    "depth-2 handoff remains conservative and should not emit ShapeSync semaphore ops yet"

@[test]
def testShapeSyncPipelineSemaphoreHandoffEmitsSynchronization : IO Unit := do
  let kernel := buildKernelM "proof_shapesync_pipeline_handoff" .SM90 #[] do
    let shared ← freshVar
    emit (.declST shared .Float32 16 16 .Row)
    let reg ← freshVar
    emit (.declRT reg .Float32 16 16 .Row)
    let global ← freshVar
    emit (.declGPtr global .Float32 "global")
    let coord ← freshVar
    emit (.constInt coord 0)
    let sem ← freshVar
    emit (.declSemaphore sem)
    pipelinedRingLoop
      { numIters := 2, depth := 1, warpSpecialized := true, useShapeSyncBarriers := true }
      (fun _ => emit (.tmaLoadAsync shared global coord coord coord coord sem))
      (fun _ => emit (.load reg shared))

  let semaphoreOps := collectSemaphoreOps kernel.body
  assertTrue (kernel.proof.shapeSyncPipelineAnalyses.size == 1)
    "ShapeSync handoff should still record one representative analysis"
  assertTrue (semaphoreOps.size == 8)
    s!"depth-1 handoff should emit two inits and six wait/arrive ops, got {repr semaphoreOps}"
  assertTrue (semaphoreOps.any (fun op => op == .Wait))
    "handoff should emit semaphore waits"
  assertTrue (semaphoreOps.any (fun op => op == .Arrive 128))
    "handoff should emit participant-counted semaphore arrives"
  assertTrue (!semaphoreOps.any (fun op => op == .Arrive 64))
    "handoff should not use the old wrong warp-group participant count"

def run : IO Unit := do
  testParticipantCountModuloGuard
  testScalarPredicateTrackingFromBuilder
  testNamedBarrierParticipantCountAcceptsWarpGroup
  testNamedBarrierParticipantCountRejectsWrongCount
  testProofFactsDerivedFromDirectKStmtTree
  testAccessDisjointnessQueuedAsStoresAreRecorded
  testAccessFactsDerivedFromDirectKStmtTree
  testOptimizationDecisionCarriesSemanticCertificate
  testAccessDisjointnessRejectsAmbiguousRace
  testOptimizationScalarRewriteCertificate
  testOptimizationRejectsInvalidScalarRewrite
  testOptimizationPredicateSimplificationCertificate
  testOptimizationAlignmentCertificateThroughKernelM
  testSyncOmissionDecisionRequiresProof
  testProofOracleProducesSemanticProofResult
  testShapeSyncBridgeConvertsClosedModuloGuard
  testShapeSyncBridgeRejectsRuntimeBoolGuard
  testShapeSyncAnalysisDerivedFromKStmtTree
  testShapeSyncCheckedNamedBarrierRecordsObligation
  testShapeSyncPipelinePreviewAnalysisRecorded
  testShapeSyncPipelineDepthTwoPreviewAnalysesRecorded
  testShapeSyncPipelineDepthTwoHandoffFallsBackToConservativeSync
  testShapeSyncPipelineSemaphoreHandoffEmitsSynchronization

def main : IO Unit := do
  run

end Tests.GPUProof
