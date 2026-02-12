/- End-to-end `mha_h100`-style forward/backward validation (2 blocks, d=64). -/
import Tyr.Torch
import Tyr.GPU.Kernels.ThunderKittensFlashAttn
import Examples.GPU.FixtureRunner

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/mha_h100_128x64"⟩
  names := #[
    "q", "k", "v", "dO",
    "expected_o", "expected_l",
    "expected_dq", "expected_dk", "expected_dv"
  ]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate mha_h100 fixtures."

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

  -- Forward reference output.
  let expectedOut := torch.nn.scaled_dot_product_attention q k v 0.0 false

  -- ThunderKittens-style L vector: L = -8 * lse for d=64.
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

  -- Backward references from PyTorch autograd.
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
  IO.println s!"Generated mha_h100 fixtures in {fixtureSpec.dir} outMean={outMean} lMean={lMean} dqMean={dqMean}"

def runOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

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

  -- Forward.
  -- Cross-check L against the existing LSE kernel path.
  let outLse := torch.zeros_like q
  let lseOut := torch.zeros #[2, 64] false (Device.CUDA 0)
  tkFlashAttnFwd2BlockLse.launch q k v outLse lseOut 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  let lFromLse : T #[2, 64] := torch.mul_scalar lseOut (-8.0)

  let out := torch.zeros_like q
  let lOut : T #[2, 64] := torch.zeros_like lseOut
  tkMhaH100Fwd2Block.launch q k v out lOut 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  -- Backward prep + backward.
  -- Keep distinct allocation expressions for mutable outputs; these FFI tensors
  -- are mutated by kernels even though constructors are pure at the Lean level.
  let dVec : T #[2, 64] := torch.mul_scalar lOut 0.0
  tkMhaH100BwdPrep2Block.launch dO out dVec 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dQ := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
  let dKPart := torch.zeros #[1, 1, 128, 128] false (Device.CUDA 0)
  let dVPartSeed := torch.ones #[1, 1, 128, 128] false (Device.CUDA 0)
  let dVPart : T #[1, 1, 128, 128] := torch.mul_scalar dVPartSeed 0.0
  tkMhaH100Bwd2BlockPartials.launch q k v dO lOut dVec dQ dKPart dVPart 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  -- Materialize post-launch views so downstream pure ops read mutated contents.
  let dKPartLive : T #[1, 1, 128, 128] := torch.add_scalar dKPart 0.0
  let dVPartLive : T #[1, 1, 128, 128] := torch.add_scalar dVPart 0.0

  -- Reduce per-(kv_tile,q_tile) partials over q_tile to recover full dK/dV.
  let dKPart6 : T #[1, 1, 2, 64, 2, 64] := torch.reshape dKPartLive #[1, 1, 2, 64, 2, 64]
  let dVPart6 : T #[1, 1, 2, 64, 2, 64] := torch.reshape dVPartLive #[1, 1, 2, 64, 2, 64]
  let dK5 : T #[1, 1, 2, 64, 64] := torch.nn.sumDim dKPart6 4 false
  let dV5 : T #[1, 1, 2, 64, 64] := torch.nn.sumDim dVPart6 4 false
  let dK : T #[1, 1, 128, 64] := torch.reshape dK5 #[1, 1, 128, 64]
  let dV : T #[1, 1, 128, 64] := torch.reshape dV5 #[1, 1, 128, 64]

  -- Validate.
  let outOk := torch.allclose expectedOut out 3e-2 3e-2
  let lOk := torch.allclose expectedL lOut 3e-2 3e-2
  let lKernelConsistent := torch.allclose lFromLse lOut 3e-2 3e-2
  let lFixtureConsistent := torch.allclose expectedL lFromLse 3e-2 3e-2
  let dqOk := torch.allclose expectedDQ dQ 8e-2 8e-2
  let dkOk := torch.allclose expectedDK dK 8e-2 8e-2
  let dvOk := torch.allclose expectedDV dV 8e-2 8e-2

  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (out - expectedOut)))
  let lMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (lOut - expectedL)))
  let lKernelMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (lOut - lFromLse)))
  let lFixtureMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (lFromLse - expectedL)))
  let dqMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dQ - expectedDQ)))
  let dkMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dK - expectedDK)))
  let dvMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dV - expectedDV)))
  let dkdvMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dK - dV)))
  IO.println s!"mha_h100 fwd_ok={outOk} l_ok={lOk} l_kernel_consistent={lKernelConsistent} l_fixture_consistent={lFixtureConsistent} dq_ok={dqOk} dk_ok={dkOk} dv_ok={dvOk} out_mae={outMae} l_mae={lMae} l_kernel_mae={lKernelMae} l_fixture_mae={lFixtureMae} dq_mae={dqMae} dk_mae={dkMae} dv_mae={dvMae} dkdv_mae={dkdvMae}"

  pure (outOk && lOk && dqOk && dkOk && dvOk)

unsafe def main (args : List String) : IO UInt32 := do
  runWithFixtures args fixtureSpec generateFixtures runOnce

end Examples.GPU

unsafe def main : List String → IO UInt32 := Examples.GPU.main
