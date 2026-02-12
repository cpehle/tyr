/- End-to-end training + benchmarking demo for ThunderKittens-style `mha_h100`.
   It optimizes q/k/v to match a fixed target attention output and can benchmark
   kernel-based training steps against a PyTorch SDPA autograd baseline. -/
import Tyr.Torch
import Tyr.GPU.Kernels.ThunderKittensFlashAttn

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels

abbrev MhaTensor := T #[1, 1, 128, 64]   -- bf16 kernel inputs/outputs
abbrev MasterTensor := T #[1, 1, 128, 64] -- fp32 optimizer state
abbrev LTensor := T #[2, 64]
abbrev PartTensor := T #[1, 1, 128, 128]

def nElems : Float := 8192.0

structure TrainSetup where
  stream : UInt64
  targetOut : MhaTensor
  targetOut32 : MasterTensor
  qInit : MasterTensor
  kInit : MasterTensor
  vInit : MasterTensor

def forwardMha (q k v : MhaTensor) (stream : UInt64) : IO (MhaTensor × LTensor) := do
  let out := torch.zeros_like q
  let lOut : LTensor := torch.zeros #[2, 64] false (Device.CUDA 0)
  tkMhaH100Fwd2Block.launch q k v out lOut 128 64 1 2 1 128 1 1 0 stream
  pure (out, lOut)

def backwardMha
    (q k v dO : MhaTensor)
    (lOut : LTensor)
    (out : MhaTensor)
    (stream : UInt64)
    : IO (MasterTensor × MasterTensor × MasterTensor) := do
  let dVec : LTensor := torch.mul_scalar lOut 0.0
  tkMhaH100BwdPrep2Block.launch dO out dVec 128 64 1 2 1 128 1 1 0 stream

  let dQ : MasterTensor := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
  let dKPart : PartTensor := torch.zeros #[1, 1, 128, 128] false (Device.CUDA 0)
  let dVPartSeed : PartTensor := torch.ones #[1, 1, 128, 128] false (Device.CUDA 0)
  let dVPart : PartTensor := torch.mul_scalar dVPartSeed 0.0
  tkMhaH100Bwd2BlockPartials.launch q k v dO lOut dVec dQ dKPart dVPart 128 64 1 2 1 128 1 1 0 stream

  -- Materialize post-launch views so downstream pure ops see mutated contents.
  let dKPartLive : PartTensor := torch.add_scalar dKPart 0.0
  let dVPartLive : PartTensor := torch.add_scalar dVPart 0.0

  -- Reduce per-(kv_tile,q_tile) partials over q_tile to recover full dK/dV.
  let dKPart6 : T #[1, 1, 2, 64, 2, 64] := torch.reshape dKPartLive #[1, 1, 2, 64, 2, 64]
  let dVPart6 : T #[1, 1, 2, 64, 2, 64] := torch.reshape dVPartLive #[1, 1, 2, 64, 2, 64]
  let dK5 : T #[1, 1, 2, 64, 64] := torch.nn.sumDim dKPart6 4 false
  let dV5 : T #[1, 1, 2, 64, 64] := torch.nn.sumDim dVPart6 4 false
  let dK : MasterTensor := torch.reshape dK5 #[1, 1, 128, 64]
  let dV : MasterTensor := torch.reshape dV5 #[1, 1, 128, 64]
  pure (dQ, dK, dV)

def lossAndGradOut (out targetOut : MhaTensor) : T #[] × MhaTensor :=
  let out32 : MasterTensor := torch.toFloat' out
  let tgt32 : MasterTensor := torch.toFloat' targetOut
  let diff : MasterTensor := out32 - tgt32
  let lossT : T #[] := torch.mul_scalar (torch.nn.meanAll (diff * diff)) 0.5
  let dO32 : MasterTensor := diff / nElems
  let dO : MhaTensor := torch.toBFloat16' dO32
  (lossT, dO)

def sgdUpdate (param grad : MasterTensor) (lr : Float) : MasterTensor :=
  param - (grad * lr)

def mkNoisyInit (targetF : MasterTensor) (noiseScale : Float) (device : Device) : IO MasterTensor := do
  let noise : MasterTensor ← torch.randn #[1, 1, 128, 64] false device
  pure (targetF + (noise * noiseScale))

def oneKernelTrainStep
    (qM kM vM : MasterTensor)
    (targetOut : MhaTensor)
    (stream : UInt64)
    (lr : Float)
    : IO (MasterTensor × MasterTensor × MasterTensor × T #[]) := do
  let q : MhaTensor := torch.toBFloat16' qM
  let k : MhaTensor := torch.toBFloat16' kM
  let v : MhaTensor := torch.toBFloat16' vM
  let (out, lOut) ← forwardMha q k v stream
  let (lossT, dO) := lossAndGradOut out targetOut
  let (dQ, dK, dV) ← backwardMha q k v dO lOut out stream
  let qM' := sgdUpdate qM dQ lr
  let kM' := sgdUpdate kM dK lr
  let vM' := sgdUpdate vM dV lr
  pure (qM', kM', vM', lossT)

def oneTorchTrainStep
    (qM kM vM : MasterTensor)
    (targetOut32 : MasterTensor)
    (lr : Float)
    : IO (MasterTensor × MasterTensor × MasterTensor × T #[]) := do
  let qLeaf := torch.autograd.set_requires_grad (torch.autograd.detach qM) true
  let kLeaf := torch.autograd.set_requires_grad (torch.autograd.detach kM) true
  let vLeaf := torch.autograd.set_requires_grad (torch.autograd.detach vM) true
  let out : MasterTensor := torch.nn.scaled_dot_product_attention qLeaf kLeaf vLeaf 0.0 false
  let diff : MasterTensor := out - targetOut32
  let lossT : T #[] := torch.mul_scalar (torch.nn.meanAll (diff * diff)) 0.5
  torch.autograd.backwardLoss lossT
  let dQ : MasterTensor := torch.autograd.grad_of qLeaf
  let dK : MasterTensor := torch.autograd.grad_of kLeaf
  let dV : MasterTensor := torch.autograd.grad_of vLeaf
  let qM' := sgdUpdate qM dQ lr
  let kM' := sgdUpdate kM dK lr
  let vM' := sgdUpdate vM dV lr
  pure (qM', kM', vM', lossT)

def makeSetup (noiseScale : Float) : IO (Option TrainSetup) := do
  if !(← torch.cuda_is_available) then
    return none

  let device := Device.CUDA 0
  let stream ← torch.cuda_current_stream

  -- Ground-truth tensors defining the training target.
  let qTargetF : MasterTensor ← torch.randn #[1, 1, 128, 64] false device
  let kTargetF : MasterTensor ← torch.randn #[1, 1, 128, 64] false device
  let vTargetF : MasterTensor ← torch.randn #[1, 1, 128, 64] false device
  let qTarget : MhaTensor := torch.toBFloat16' qTargetF
  let kTarget : MhaTensor := torch.toBFloat16' kTargetF
  let vTarget : MhaTensor := torch.toBFloat16' vTargetF
  let (targetOut, _) ← forwardMha qTarget kTarget vTarget stream

  let qInit ← mkNoisyInit qTargetF noiseScale device
  let kInit ← mkNoisyInit kTargetF noiseScale device
  let vInit ← mkNoisyInit vTargetF noiseScale device
  let targetOut32 : MasterTensor := torch.toFloat' targetOut
  let _ ← torch.cuda_synchronize

  return some {
    stream, targetOut, targetOut32, qInit, kInit, vInit
  }

private def parseArgNat (args : List String) (flag : String) (default : Nat) : Nat := Id.run do
  let rec loop (acc : Nat) (xs : List String) : Nat :=
    match xs with
    | k :: v :: rest =>
        if k == flag then
          let acc :=
            match v.toNat? with
            | some n => n
            | none => acc
          loop acc rest
        else
          loop acc (v :: rest)
    | _ => acc
  loop default args

private def parseFloatLit? (s : String) : Option Float :=
  match s.splitOn "." with
  | [whole] =>
      whole.toNat?.map (·.toFloat)
  | [whole, frac] =>
      match whole.toNat?, frac.toNat? with
      | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (w.toFloat + f.toFloat / denom)
      | _, _ => none
  | _ => none

private def parseArgFloat (args : List String) (flag : String) (default : Float) : Float := Id.run do
  let rec loop (acc : Float) (xs : List String) : Float :=
    match xs with
    | k :: v :: rest =>
        if k == flag then
          let acc :=
            match parseFloatLit? v with
            | some x => x
            | none => acc
          loop acc rest
        else
          loop acc (v :: rest)
    | _ => acc
  loop default args

def runTrainKernel (steps : Nat) (lr : Float) (noiseScale : Float) (logEvery : Nat) : IO Bool := do
  if steps == 0 then
    IO.eprintln "--steps must be > 0"
    return false

  let some setup ← makeSetup noiseScale
    | IO.eprintln "CUDA is not available on this host."; return false

  let mut qM := torch.autograd.clone setup.qInit
  let mut kM := torch.autograd.clone setup.kInit
  let mut vM := torch.autograd.clone setup.vInit

  let mut firstLoss : Float := 0.0
  let mut lastLoss : Float := 0.0

  for step in [:steps] do
    let (qN, kN, vN, lossT) ← oneKernelTrainStep qM kM vM setup.targetOut setup.stream lr
    qM := qN; kM := kN; vM := vN

    let mustLog := step == 0 || step + 1 == steps || (logEvery > 0 && step % logEvery == 0)
    if mustLog then
      let _ ← torch.cuda_synchronize
      let loss := torch.nn.item lossT
      if step == 0 then
        firstLoss := loss
      if step + 1 == steps then
        lastLoss := loss
      IO.println s!"mha_h100_train step={step+1}/{steps} loss={loss}"

  let relImprovement :=
    if firstLoss > 0.0 then (firstLoss - lastLoss) / firstLoss else 0.0
  let ok : Bool := if lastLoss < firstLoss then true else false
  IO.println s!"mha_h100_train init_loss={firstLoss} final_loss={lastLoss} rel_improvement={relImprovement} ok={ok}"
  pure ok

def benchKernel (setup : TrainSetup) (warmup benchIters : Nat) (lr : Float) : IO Float := do
  let mut qM := torch.autograd.clone setup.qInit
  let mut kM := torch.autograd.clone setup.kInit
  let mut vM := torch.autograd.clone setup.vInit

  for _ in [:warmup] do
    let (qN, kN, vN, _) ← oneKernelTrainStep qM kM vM setup.targetOut setup.stream lr
    qM := qN; kM := kN; vM := vN
  let _ ← torch.cuda_synchronize

  -- Reinitialize to avoid timing warmup-mutated state.
  qM := torch.autograd.clone setup.qInit
  kM := torch.autograd.clone setup.kInit
  vM := torch.autograd.clone setup.vInit

  let t0 ← IO.monoNanosNow
  for _ in [:benchIters] do
    let (qN, kN, vN, _) ← oneKernelTrainStep qM kM vM setup.targetOut setup.stream lr
    qM := qN; kM := kN; vM := vN
  let _ ← torch.cuda_synchronize
  let t1 ← IO.monoNanosNow

  let elapsedMs := (t1 - t0).toFloat / 1000000.0
  pure (elapsedMs / benchIters.toFloat)

def benchTorch (setup : TrainSetup) (warmup benchIters : Nat) (lr : Float) : IO Float := do
  let mut qM := torch.autograd.clone setup.qInit
  let mut kM := torch.autograd.clone setup.kInit
  let mut vM := torch.autograd.clone setup.vInit

  for _ in [:warmup] do
    let (qN, kN, vN, _) ← oneTorchTrainStep qM kM vM setup.targetOut32 lr
    qM := qN; kM := kN; vM := vN
  let _ ← torch.cuda_synchronize

  qM := torch.autograd.clone setup.qInit
  kM := torch.autograd.clone setup.kInit
  vM := torch.autograd.clone setup.vInit

  let t0 ← IO.monoNanosNow
  for _ in [:benchIters] do
    let (qN, kN, vN, _) ← oneTorchTrainStep qM kM vM setup.targetOut32 lr
    qM := qN; kM := kN; vM := vN
  let _ ← torch.cuda_synchronize
  let t1 ← IO.monoNanosNow

  let elapsedMs := (t1 - t0).toFloat / 1000000.0
  pure (elapsedMs / benchIters.toFloat)

def runBenchmark
    (warmup benchIters : Nat)
    (lr noiseScale : Float)
    (compareTorch : Bool)
    : IO Bool := do
  if benchIters == 0 then
    IO.eprintln "--bench-iters must be > 0"
    return false

  let some setup ← makeSetup noiseScale
    | IO.eprintln "CUDA is not available on this host."; return false

  let kernelMs ← benchKernel setup warmup benchIters lr
  let kernelStepsPerSec := 1000.0 / kernelMs
  IO.println s!"mha_h100_bench kernel_ms_per_step={kernelMs} kernel_steps_per_sec={kernelStepsPerSec} warmup={warmup} bench_iters={benchIters}"

  if compareTorch then
    let torchMs ← benchTorch setup warmup benchIters lr
    let torchStepsPerSec := 1000.0 / torchMs
    let kernelSpeedupVsTorch := torchMs / kernelMs
    IO.println s!"mha_h100_bench torch_ms_per_step={torchMs} torch_steps_per_sec={torchStepsPerSec} kernel_speedup_vs_torch={kernelSpeedupVsTorch}"

  pure true

def main (args : List String) : IO UInt32 := do
  -- Keep compatibility with scripts/gpu/run_e2e_kernel.sh flow.
  if args.contains "--gen-only" then
    return 0

  let steps := parseArgNat args "--steps" 30
  let lr := parseArgFloat args "--lr" 200.0
  let noiseScale := parseArgFloat args "--noise" 0.5
  let logEvery := parseArgNat args "--log-every" 5
  let warmup := parseArgNat args "--warmup" 10
  let benchIters := parseArgNat args "--bench-iters" 200
  let benchmark := args.contains "--benchmark"
  let compareTorch := !args.contains "--no-compare"

  let ok ←
    if benchmark then
      runBenchmark warmup benchIters lr noiseScale compareTorch
    else
      runTrainKernel steps lr noiseScale logEvery
  pure (if ok then 0 else 1)

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
