/- End-to-end FlashAttention validation for 2-block forward + forward-with-LSE. -/
import Tyr.Torch
import Tyr.GPU.Kernels.ThunderKittensFlashAttn
import Examples.GPU.FixtureRunner

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/flashattn128x64"⟩
  names := #["q", "k", "v", "expected_o", "expected_lse"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate flash attention fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device := Device.CUDA 0

  let qf ← torch.randn #[1, 1, 128, 64] false device
  let kf ← torch.randn #[1, 1, 128, 64] false device
  let vf ← torch.randn #[1, 1, 128, 64] false device

  let q := torch.toBFloat16' qf
  let k := torch.toBFloat16' kf
  let v := torch.toBFloat16' vf
  -- Quantize inputs to bf16, then compute references in float32.
  let q32 := torch.toFloat' q
  let k32 := torch.toFloat' k
  let v32 := torch.toFloat' v

  let kT : T #[1, 1, 64, 128] := torch.nn.transpose k32 2 3
  let scores : T #[1, 1, 128, 128] := torch.nn.bmm4d q32 kT
  let scaled : T #[1, 1, 128, 128] := scores / 8.0
  let probs : T #[1, 1, 128, 128] := torch.nn.softmax_dim scaled (-1)
  let expectedOut32 : T #[1, 1, 128, 64] := torch.nn.bmm4d probs v32
  let expectedOut := torch.toBFloat16' expectedOut32

  let expScores : T #[1, 1, 128, 128] := torch.nn.exp scaled
  let sumExp : T #[1, 1, 128] := torch.nn.sumDim expScores 3 false
  let expectedLse3 : T #[1, 1, 128] := torch.nn.log sumExp
  let expectedLse : T #[2, 64] := torch.reshape expectedLse3 #[2, 64]

  torch.data.saveTensor q (fixtureFile "q").toString
  torch.data.saveTensor k (fixtureFile "k").toString
  torch.data.saveTensor v (fixtureFile "v").toString
  torch.data.saveTensor expectedOut (fixtureFile "expected_o").toString
  torch.data.saveTensor expectedLse (fixtureFile "expected_lse").toString

  let outMean := torch.nn.item (torch.nn.meanAll expectedOut)
  let lseMean := torch.nn.item (torch.nn.meanAll expectedLse)
  IO.println s!"Generated flash attention fixtures in {fixtureSpec.dir} outMean={outMean} lseMean={lseMean}"

def runOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let q ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "q").toString
  let k ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "k").toString
  let v ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "v").toString
  let expectedOut ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_o").toString
  let expectedLse ← torch.data.loadTensor #[2, 64] (fixtureFile "expected_lse").toString

  let outFwd := torch.zeros_like q
  let outFwdLse := torch.zeros_like q
  let lseOut := torch.zeros #[2, 64] false (Device.CUDA 0)
  let stream ← torch.cuda_current_stream

  -- grid=(1,2,1): one CTA per 64-row query tile.
  tkFlashAttnFwd2Block.launch q k v outFwd 128 64 1 2 1 128 1 1 0 stream
  tkFlashAttnFwd2BlockLse.launch q k v outFwdLse lseOut 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let outOk := torch.allclose expectedOut outFwd 3e-2 3e-2
  let outLseKernelOk := torch.allclose expectedOut outFwdLse 3e-2 3e-2
  let lseOk := torch.allclose expectedLse lseOut 3e-2 3e-2

  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (outFwd - expectedOut)))
  let outMaxErr := torch.nn.item (torch.nn.maxAll (torch.nn.abs (outFwd - expectedOut)))
  let lseMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (lseOut - expectedLse)))
  let lseMaxErr := torch.nn.item (torch.nn.maxAll (torch.nn.abs (lseOut - expectedLse)))

  IO.println s!"flashattn2block fwd_ok={outOk} fwd_lse_ok={outLseKernelOk} lse_ok={lseOk} out_mae={outMae} out_max={outMaxErr} lse_mae={lseMae} lse_max={lseMaxErr}"
  pure (outOk && outLseKernelOk && lseOk)

unsafe def main (args : List String) : IO UInt32 := do
  runWithFixtures args fixtureSpec generateFixtures runOnce

end Examples.GPU

unsafe def main : List String → IO UInt32 := Examples.GPU.main
