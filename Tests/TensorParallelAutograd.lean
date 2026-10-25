import Tyr.Distributed
import Lean

open Lean torch

private def check (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw <| IO.userError message

private def traceStage (label : String) : IO Unit := do
  let stream ← IO.getStderr
  stream.putStrLn s!"TP TEST: {label}"
  stream.flush

private def tensor (shape : Shape) (values : Array Float) (device : Device)
    (trainable : Bool := false) : T shape :=
  autograd.set_requires_grad (autograd.detach ((reshape (data.fromFloatArray values) shape).to device)) trainable

private def values {s : Shape} (x : T s) : IO (Array Float) :=
  data.tensorToFloatArray' (nn.eraseShape (autograd.detach x))

private def equal {s : Shape} (label : String) (x : T s) (expected : Array Float) : IO Unit := do
  traceStage s!"check {label}"
  let actual ← values x
  check (actual.size == expected.size) s!"{label}: wrong element count"
  for i in [:expected.size] do
    let a := actual[i]!
    let e := expected[i]!
    check (!a.isNaN && !a.isInf && (a-e).abs <= 1e-6)
      s!"{label}[{i}]: got {a}, expected {e}"

private def rejected (action : IO α) (expected : String := "") : IO Unit := do
  let failure ← try let _ ← action; pure none catch error => pure (some error.toString)
  let some failure := failure | throw <| IO.userError "expected rejection"
  check (expected.isEmpty || (failure.splitOn expected).length > 1)
    s!"unexpected rejection: {failure}"

private def copyRule (rank : Nat) (device : Device) : IO Unit := do
  let x := tensor #[2] #[2,3] device true
  let output ← dist.copyToTensorParallel x
  equal "copy forward" output #[2,3]
  let upstream := tensor #[2] (if rank == 0 then #[1,2] else #[3,5]) device
  -- Passing the upstream tensor directly also checks backward does not mutate it.
  traceStage "explicit-upstream backward"
  autograd.backward output upstream
  let gradient := autograd.grad_of x
  equal "copy backward SUM" gradient #[4,7]
  equal "copy upstream unchanged" upstream (if rank == 0 then #[1,2] else #[3,5])
  equal "copy input unchanged" x #[2,3]

private def reduceRule (rank : Nat) (device : Device) : IO Unit := do
  let original : Array Float := if rank == 0 then #[1,2] else #[3,5]
  let x := tensor #[2] original device true
  let output ← dist.reduceFromTensorParallel x
  equal "reduce forward SUM" output #[4,7]
  equal "reduce forward input unchanged" x original
  let upstreamValues : Array Float := if rank == 0 then #[2,3] else #[5,7]
  let upstream := tensor #[2] upstreamValues device
  traceStage "explicit-upstream backward"
  autograd.backward output upstream
  let gradient := autograd.grad_of x
  equal "reduce backward identity, not SUM" gradient upstreamValues
  equal "reduce upstream unchanged" upstream upstreamValues

private def ownedStorage (rank : Nat) (device : Device) : IO Unit := do
  let original : Array Float := if rank == 0 then #[1,2] else #[3,5]
  let input := tensor #[2] original device
  let copied ← dist.copyToTensorParallel input
  -- Mutate only the returned tensor using an existing explicit collective.
  dist.allReduce copied .sum
  equal "copy output mutation" copied #[4,7]
  equal "copy owns storage" input original
  let reduced ← dist.reduceFromTensorParallel input
  dist.allReduce reduced .sum
  equal "reduce output mutation" reduced #[8,14]
  equal "reduce owns storage" input original

private def noncontiguous (rank : Nat) (device : Device) : IO Unit := do
  let original := tensor #[2,2] #[1,2,3,4] device true
  let view := nn.transpose original 0 1
  let copied ← dist.copyToTensorParallel view
  equal "noncontiguous copy output" copied #[1,3,2,4]
  let upstream := nn.transpose (tensor #[2,2] (if rank == 0 then #[1,2,3,4] else #[5,6,7,8]) device) 0 1
  traceStage "noncontiguous copy backward"
  autograd.backward copied upstream
  equal "noncontiguous copy gradient" (autograd.grad_of original) #[6,8,10,12]
  equal "noncontiguous upstream unchanged" upstream (if rank == 0 then #[1,3,2,4] else #[5,7,6,8])
  dist.allReduce (autograd.detach copied) .sum
  equal "noncontiguous copy owns storage" original #[1,2,3,4]
  let originalValues : Array Float := if rank == 0 then #[1,2,3,4] else #[5,6,7,8]
  let original := tensor #[2,2] originalValues device true
  let reduced ← dist.reduceFromTensorParallel (nn.transpose original 0 1)
  equal "noncontiguous reduce output" reduced #[6,10,8,12]
  let gradientValues : Array Float := if rank == 0 then #[2,3,5,7] else #[11,13,17,19]
  let upstream := nn.transpose (tensor #[2,2] gradientValues device) 0 1
  traceStage "noncontiguous reduce backward"
  autograd.backward reduced upstream
  equal "noncontiguous reduce gradient" (autograd.grad_of original) gradientValues
  dist.allReduce (autograd.detach reduced) .sum
  equal "noncontiguous reduce owns storage" original originalValues

private def denseComposition (rank : Nat) (device : Device) : IO Unit := do
  let input := tensor #[1,2] #[2,3] device true
  let column := tensor #[2,1] (if rank == 0 then #[1,2] else #[3,4]) device true
  let row := tensor #[1,2] (if rank == 0 then #[5,6] else #[7,8]) device true
  let copied ← dist.copyToTensorParallel input
  let hidden := nn.mm copied column
  let output ← dist.reduceFromTensorParallel (nn.mm hidden row)
  equal "toy output" output #[166,192]
  let upstream := tensor #[1,2] #[2,-1] device
  autograd.backwardLoss (nn.sumAll (output * upstream))
  equal "toy input gradient" (autograd.grad_of input) #[22,32]
  equal "toy column gradient" (autograd.grad_of column) (if rank == 0 then #[8,12] else #[12,18])
  equal "toy row gradient" (autograd.grad_of row) (if rank == 0 then #[16,-8] else #[36,-18])
  -- Independent ordinary full-matrix graph, without any distributed mapping.
  let referenceInput := tensor #[1,2] #[2,3] device true
  let referenceColumn := tensor #[2,2] #[1,3,2,4] device true
  let referenceRow := tensor #[2,2] #[5,6,7,8] device true
  let reference := nn.mm (nn.mm referenceInput referenceColumn) referenceRow
  equal "ordinary toy output" reference #[166,192]
  autograd.backwardLoss (nn.sumAll (reference * upstream))
  equal "ordinary toy input gradient" (autograd.grad_of referenceInput) #[22,32]
  equal "ordinary toy column gradient" (autograd.grad_of referenceColumn) #[8,12,12,18]
  equal "ordinary toy row gradient" (autograd.grad_of referenceRow) #[16,-8,36,-18]

private def staleGraphs (backend address : String) (port rank : Nat) (device : Device) : IO Unit := do
  let originalGeneration ← dist.getProcessGroupGeneration
  check (originalGeneration > 0) "active group generation must be positive"
  let make := fun (reduce : Bool) => do
    let x := tensor #[2] #[2,3] device true
    let y ← if reduce then dist.reduceFromTensorParallel x else dist.copyToTensorParallel x
    pure (nn.sumAll y)
  let destroyedCopy ← make false
  let destroyedReduce ← make true
  let replacedCopy ← make false
  let replacedReduce ← make true
  dist.barrier
  dist.destroyProcessGroup
  traceStage "reject generation query without group"
  rejected dist.getProcessGroupGeneration
  let stale := "Tensor-parallel autograd process-group generation is no longer active"
  traceStage "reject destroyedCopy"
  rejected (autograd.backwardLoss destroyedCopy) stale
  traceStage "reject destroyedReduce"
  rejected (autograd.backwardLoss destroyedReduce) stale
  traceStage "reject copy without group"
  rejected (dist.copyToTensorParallel (tensor #[1] #[1] device))
  traceStage "reject reduce without group"
  rejected (dist.reduceFromTensorParallel (tensor #[1] #[1] device))
  dist.initProcessGroup backend address (port+1).toUInt64 rank.toUInt64 2
  check ((← dist.getProcessGroupGeneration) > originalGeneration) "replacement group did not advance native generation"
  traceStage "reject replacedCopy"
  rejected (autograd.backwardLoss replacedCopy) stale
  traceStage "reject replacedReduce"
  rejected (autograd.backwardLoss replacedReduce) stale
  -- Rejection of stale graphs must not poison the valid replacement group.
  copyRule rank device
  reduceRule rank device

private def readyGate (rank : Nat) (device : Device) : IO Unit := do
  let some directory ← IO.getEnv "TYR_TP_TEST_GATE" | return
  -- Materialize a real device allocation before the harness attributes GPU PIDs.
  equal "device readiness" (tensor #[1] #[1] device) #[1]
  dist.barrier
  let root : System.FilePath := directory
  let release := root / "release"
  check (!(← release.pathExists)) "test gate already released"
  let pending := root / s!"rank-{rank}.ready.tmp"
  IO.FS.writeFile pending (Json.mkObj [("rank",toJson rank),("ready",toJson true)]).pretty
  IO.FS.rename pending (root / s!"rank-{rank}.ready.json")
  let deadline := (← IO.monoMsNow) + 120000
  while !(← release.pathExists) do
    check ((← IO.monoMsNow) < deadline) "test gate deadline"
    IO.sleep 10

def main (args : List String) : IO Unit := do
  traceStage "main entry"
  let [backend,address,portText,rankText,output] := args |
    throw <| IO.userError "usage: BACKEND MASTER_ADDR PORT RANK OUTPUT_JSON"
  let some port := portText.toNat? | throw <| IO.userError "invalid port"
  let some rank := rankText.toNat? | throw <| IO.userError "invalid rank"
  check (port > 0 && port < 65535 && rank < 2 && (backend == "gloo" || backend == "nccl")) "invalid topology"
  let device := if backend == "nccl" then Device.CUDA 0 else Device.CPU
  traceStage "set CUDA device"
  if backend == "nccl" then dist.setCudaDevice 0
  traceStage "enable gradients"
  autograd.set_grad_enabled true
  traceStage "reject generation query without group"
  rejected dist.getProcessGroupGeneration
  traceStage "reject copy without group"
  rejected (dist.copyToTensorParallel (tensor #[1] #[1] device))
  traceStage "reject reduce without group"
  rejected (dist.reduceFromTensorParallel (tensor #[1] #[1] device))
  traceStage "reject unsupported backend"
  rejected (dist.initProcessGroup "unsupported-test-backend" address port.toUInt64 rank.toUInt64 2)
    "Unsupported backend"
  check (!(← dist.isInitialized)) "failed initialization exposed a group"
  traceStage "initialize valid process group"
  dist.initProcessGroup backend address port.toUInt64 rank.toUInt64 2
  try
    traceStage "ready gate"
    readyGate rank device
    traceStage "begin copyRule"
    copyRule rank device
    traceStage "copy identity/SUM and upstream nonmutation PASS"
    traceStage "begin reduceRule"
    reduceRule rank device
    traceStage "reduce SUM/identity and input nonmutation PASS"
    traceStage "begin ownedStorage"
    ownedStorage rank device
    traceStage "owned output storage PASS"
    traceStage "begin noncontiguous"
    noncontiguous rank device
    traceStage "noncontiguous inputs/upstreams and storage ownership PASS"
    traceStage "begin denseComposition"
    denseComposition rank device
    traceStage "column/row composition and ordinary gradient reference PASS"
    traceStage "begin staleGraphs"
    staleGraphs backend address port rank device
    traceStage "destroyed/replaced group graph rejection and fresh graph recovery PASS"
    dist.barrier
  finally
    if ← dist.isInitialized then dist.destroyProcessGroup
  IO.FS.writeFile output (Json.mkObj [
    ("status",toJson "passed"),("rank",toJson rank),("backend",toJson backend),
    ("absolute_tolerance",toJson (1e-6 : Float)),
    ("checks",toJson #["copy_identity_backward_sum","copy_upstream_unchanged",
      "reduce_sum_backward_identity","reduce_input_unchanged","owned_output_storage",
      "noncontiguous_inputs_and_upstreams",
      "column_row_analytic_gradients","ordinary_dense_reference",
      "failed_initialization_then_valid_group",
      "destroyed_graph_rejected","replaced_graph_rejected","fresh_group_recovers"]) ]).pretty
