/- End-to-end rotary validation:
   generate deterministic input/reference tensors, launch the kernel, compare outputs. -/
import Tyr.Torch
import Tyr.GPU.Kernels.Rotary
import Examples.GPU.FixtureRunner

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels.Rotary

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/rotary64"⟩
  names := #["x", "sin", "cos", "expected"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate rotary fixtures."

  IO.FS.createDirAll fixtureSpec.dir

  let device := Device.CUDA 0
  let x ← torch.rand #[64, 64] false device

  let (cosCpu, sinCpu) := torch.rotary.computeFreqsPure (64 : UInt64) (64 : UInt64) 10000.0
  let cos : T #[64, 32] := cosCpu.to device
  let sin : T #[64, 32] := sinCpu.to device

  let x4 : T #[1, 64, 1, 64] := torch.reshape x #[1, 64, 1, 64]
  let expected4 := torch.rotary.applyRotaryEmb x4 cos sin
  let expected : T #[64, 64] := torch.reshape expected4 #[64, 64]

  torch.data.saveTensor x (fixtureFile "x").toString
  torch.data.saveTensor sin (fixtureFile "sin").toString
  torch.data.saveTensor cos (fixtureFile "cos").toString
  torch.data.saveTensor expected (fixtureFile "expected").toString

  let xMean := torch.nn.item (torch.nn.meanAll x)
  let eMean := torch.nn.item (torch.nn.meanAll expected)
  IO.println s!"Generated rotary fixtures in {fixtureSpec.dir} xMean={xMean} expectedMean={eMean}"

def runOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let x ← torch.data.loadTensor #[64, 64] (fixtureFile "x").toString
  let sin ← torch.data.loadTensor #[64, 32] (fixtureFile "sin").toString
  let cos ← torch.data.loadTensor #[64, 32] (fixtureFile "cos").toString
  let expected ← torch.data.loadTensor #[64, 64] (fixtureFile "expected").toString

  let output := torch.zeros_like x
  let stream ← torch.cuda_current_stream

  -- grid=(1,1,1), block=(128,1,1), sharedMem=0
  rotaryFwd.launch x sin cos output 64 64 1 1 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let ok := torch.allclose expected output 1e-4 1e-4
  let mae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (output - expected)))
  let outMean := torch.nn.item (torch.nn.meanAll output)
  let expMean := torch.nn.item (torch.nn.meanAll expected)
  IO.println s!"rotary allclose={ok} mae={mae} outMean={outMean} expectedMean={expMean}"
  pure ok

unsafe def main (args : List String) : IO UInt32 := do
  runWithFixtures args fixtureSpec generateFixtures runOnce

end Examples.GPU

unsafe def main : List String → IO UInt32 := Examples.GPU.main
