/- End-to-end rotary validation:
   generate deterministic input/reference tensors, launch the kernel, compare outputs. -/
import Tyr.Torch
import Tyr.GPU.Kernels.Rotary
import Examples.GPU.Parity
import Examples.GPU.FixtureRunner

namespace Examples.GPU.RunRotary

open torch
open Tyr.GPU.Kernels.Rotary

def suiteName : String := "rotary"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/rotary64"⟩
  names := #["x", "sin", "cos", "expected"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
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
  if !(← requireCuda suiteName) then
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

  let check := compareTensors "rotary.output" expected output 1e-4 1e-4
  let outMean := torch.nn.item (torch.nn.meanAll output)
  let expMean := torch.nn.item (torch.nn.meanAll expected)
  logTensorCheck check
  IO.println s!"rotary output_mean={outMean} expected_mean={expMean}"
  pure check.ok

def main (args : List String) : IO UInt32 := do
  runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunRotary
