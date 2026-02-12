/- End-to-end ThunderKittens-style layernorm validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.ThunderKittensLayerNorm
import Examples.GPU.FixtureRunner

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/layernorm64x1024"⟩
  names := #["x", "residual", "weight", "bias", "expected_out", "expected_resid"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate layernorm fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device := Device.CUDA 0
  let eps : Float := 1.0e-5

  let x ← torch.rand #[1, 64, 1024] false device
  let residual ← torch.rand #[1, 64, 1024] false device
  let weight ← torch.rand #[1024] false device
  let bias ← torch.rand #[1024] false device

  let expectedResid := x + residual
  let expectedOut := torch.nn.layer_norm expectedResid weight bias eps

  torch.data.saveTensor x (fixtureFile "x").toString
  torch.data.saveTensor residual (fixtureFile "residual").toString
  torch.data.saveTensor weight (fixtureFile "weight").toString
  torch.data.saveTensor bias (fixtureFile "bias").toString
  torch.data.saveTensor expectedOut (fixtureFile "expected_out").toString
  torch.data.saveTensor expectedResid (fixtureFile "expected_resid").toString

  let outMean := torch.nn.item (torch.nn.meanAll expectedOut)
  let residMean := torch.nn.item (torch.nn.meanAll expectedResid)
  IO.println s!"Generated layernorm fixtures in {fixtureSpec.dir} outMean={outMean} residMean={residMean}"

def runOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let x ← torch.data.loadTensor #[1, 64, 1024] (fixtureFile "x").toString
  let residual ← torch.data.loadTensor #[1, 64, 1024] (fixtureFile "residual").toString
  let weight ← torch.data.loadTensor #[1024] (fixtureFile "weight").toString
  let bias ← torch.data.loadTensor #[1024] (fixtureFile "bias").toString
  let expectedOut ← torch.data.loadTensor #[1, 64, 1024] (fixtureFile "expected_out").toString
  let expectedResid ← torch.data.loadTensor #[1, 64, 1024] (fixtureFile "expected_resid").toString

  let out := torch.zeros_like x
  -- Use a distinct source expression to prevent CSE from aliasing outputs.
  let outResid := torch.zeros_like residual
  let stream ← torch.cuda_current_stream

  -- Single-warp launch keeps numerical behavior deterministic for this kernel.
  tkLayerNorm.launch x residual weight bias out outResid 1 1 1 32 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let outOk := torch.allclose expectedOut out 5e-3 5e-3
  let residOk := torch.allclose expectedResid outResid 1e-5 1e-5

  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (out - expectedOut)))
  let residMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (outResid - expectedResid)))
  let outMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (out - expectedOut)))
  IO.println s!"layernorm out_allclose={outOk} resid_allclose={residOk} out_mae={outMae} resid_mae={residMae} out_max={outMax}"
  pure (outOk && residOk)

unsafe def main (args : List String) : IO UInt32 := do
  runWithFixtures args fixtureSpec generateFixtures runOnce

end Examples.GPU

unsafe def main : List String → IO UInt32 := Examples.GPU.main
