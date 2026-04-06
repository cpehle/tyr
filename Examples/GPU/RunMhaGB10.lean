/- End-to-end GB10/Blackwell MHA validation using the reduced 2-block path. -/
import Tyr.Torch
import Tyr.GPU.Kernels.MhaGB10
import Examples.GPU.Parity
import Examples.GPU.FixtureRunner

namespace Examples.GPU.RunMhaGB10

open torch
open Tyr.GPU.Kernels

def suiteName : String := "mha_gb10"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/mha_gb10_128x64"⟩
  names := #[
    "q", "k", "v", "dO",
    "expected_o", "expected_l",
    "expected_dq", "expected_dk", "expected_dv"
  ]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate mha_gb10 fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device := Device.CUDA 0

  let qf ← torch.randn #[1, 1, 128, 64] false device
  let kf ← torch.randn #[1, 1, 128, 64] false device
  let vf ← torch.randn #[1, 1, 128, 64] false device
  let dOf ← torch.randn #[1, 1, 128, 64] false device

  let q := torch.toBFloat16' qf
  let k := torch.toBFloat16' kf
  let v := torch.toBFloat16' vf
  let dO := torch.toBFloat16' dOf

  let expectedOut := torch.nn.scaled_dot_product_attention q k v 0.0 false

  let q32 := torch.toFloat' q
  let k32 := torch.toFloat' k
  let kT : T #[1, 1, 64, 128] := torch.nn.transpose k32 2 3
  let scores : T #[1, 1, 128, 128] := torch.nn.bmm4d q32 kT
  let scaled : T #[1, 1, 128, 128] := scores / 8.0
  let expScores : T #[1, 1, 128, 128] := torch.nn.exp scaled
  let sumExp : T #[1, 1, 128] := torch.nn.sumDim expScores 3 false
  let lse3 : T #[1, 1, 128] := torch.nn.log sumExp
  let expectedL3 : T #[1, 1, 128] := torch.mul_scalar lse3 (-8.0)
  let expectedL : T #[2, 64] := torch.reshape expectedL3 #[2, 64]

  let qRef := torch.autograd.set_requires_grad q true
  let kRef := torch.autograd.set_requires_grad k true
  let vRef := torch.autograd.set_requires_grad v true
  let outRef := torch.nn.scaled_dot_product_attention qRef kRef vRef 0.0 false
  torch.autograd.backward outRef dO
  let expectedDQ := torch.toFloat' (torch.autograd.grad_of qRef)
  let expectedDK := torch.toFloat' (torch.autograd.grad_of kRef)
  let expectedDV := torch.toFloat' (torch.autograd.grad_of vRef)

  torch.data.saveTensor q (fixtureFile "q").toString
  torch.data.saveTensor k (fixtureFile "k").toString
  torch.data.saveTensor v (fixtureFile "v").toString
  torch.data.saveTensor dO (fixtureFile "dO").toString
  torch.data.saveTensor expectedOut (fixtureFile "expected_o").toString
  torch.data.saveTensor expectedL (fixtureFile "expected_l").toString
  torch.data.saveTensor expectedDQ (fixtureFile "expected_dq").toString
  torch.data.saveTensor expectedDK (fixtureFile "expected_dk").toString
  torch.data.saveTensor expectedDV (fixtureFile "expected_dv").toString

  let outMean := torch.nn.item (torch.nn.meanAll expectedOut)
  let lMean := torch.nn.item (torch.nn.meanAll expectedL)
  let dqMean := torch.nn.item (torch.nn.meanAll expectedDQ)
  IO.println s!"Generated mha_gb10 fixtures in {fixtureSpec.dir} outMean={outMean} lMean={lMean} dqMean={dqMean}"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  if !(← isBlackwellFamily) then
    IO.println s!"[skip] {suiteName}: requires TYR_GPU_FAMILY=BLACKWELL"
    return true

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let stream ← torch.cuda_current_stream

  let q ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "q").toString
  let k ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "k").toString
  let v ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "v").toString
  let dO ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "dO").toString
  let expectedOut ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_o").toString
  let expectedL ← torch.data.loadTensor #[2, 64] (fixtureFile "expected_l").toString
  let expectedDQ ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dq").toString
  let expectedDK ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dk").toString
  let expectedDV ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dv").toString

  let outLse := torch.zeros_like q
  let lseOut := torch.zeros #[2, 64] false (Device.CUDA 0)
  tkFlashAttnGb10Fwd2BlockLse.launch q k v outLse lseOut 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  let lFromLse : T #[2, 64] := torch.mul_scalar lseOut (-8.0)

  let out := torch.zeros_like q
  let lOut : T #[2, 64] := torch.zeros_like lseOut
  tkMhaGb10Fwd2Block.launch q k v out lOut 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dVec : T #[2, 64] := torch.mul_scalar lOut 0.0
  tkMhaGb10BwdPrep2Block.launch dO out dVec 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dQ := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
  let dKPart := torch.zeros #[1, 1, 128, 128] false (Device.CUDA 0)
  let dVPartSeed := torch.ones #[1, 1, 128, 128] false (Device.CUDA 0)
  let dVPart : T #[1, 1, 128, 128] := torch.mul_scalar dVPartSeed 0.0
  tkMhaGb10Bwd2BlockPartials.launch q k v dO lOut dVec dQ dKPart dVPart 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  let dKPartLive : T #[1, 1, 128, 128] := torch.add_scalar dKPart 0.0
  let dVPartLive : T #[1, 1, 128, 128] := torch.add_scalar dVPart 0.0

  let dKPart6 : T #[1, 1, 2, 64, 2, 64] := torch.reshape dKPartLive #[1, 1, 2, 64, 2, 64]
  let dVPart6 : T #[1, 1, 2, 64, 2, 64] := torch.reshape dVPartLive #[1, 1, 2, 64, 2, 64]
  let dK5 : T #[1, 1, 2, 64, 64] := torch.nn.sumDim dKPart6 4 false
  let dV5 : T #[1, 1, 2, 64, 64] := torch.nn.sumDim dVPart6 4 false
  let dK : T #[1, 1, 128, 64] := torch.reshape dK5 #[1, 1, 128, 64]
  let dV : T #[1, 1, 128, 64] := torch.reshape dV5 #[1, 1, 128, 64]

  let outCheck := compareTensors "mha_gb10.forward" expectedOut out 3e-2 3e-2
  let lCheck := compareTensors "mha_gb10.l" expectedL lOut 3e-2 3e-2
  let lKernelCheck := compareTensors "mha_gb10.l_vs_lse_kernel" lFromLse lOut 3e-2 3e-2
  let lFixtureCheck := compareTensors "mha_gb10.lse_kernel_vs_fixture" expectedL lFromLse 3e-2 3e-2
  let dqCheck := compareTensors "mha_gb10.dq" expectedDQ dQ 3e-2 3e-2
  let dkCheck := compareTensors "mha_gb10.dk" expectedDK dK 3e-2 3e-2
  let dvCheck := compareTensors "mha_gb10.dv" expectedDV dV 3e-2 3e-2
  for check in #[outCheck, lCheck, lKernelCheck, lFixtureCheck, dqCheck, dkCheck, dvCheck] do
    logTensorCheck check

  pure (outCheck.ok && lCheck.ok && dqCheck.ok && dkCheck.ok && dvCheck.ok)

def main (args : List String) : IO UInt32 := do
  runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunMhaGB10
