/- End-to-end rotary validation:
   generate deterministic input/reference tensors, launch the kernel, compare outputs. -/
import Tyr.Torch
import Tyr.GPU.Kernels.Rotary

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels.Rotary

def fixtureDir : System.FilePath := ⟨"data/gpu_fixtures/rotary64"⟩

def fixturePath (name : String) : System.FilePath :=
  fixtureDir / s!"{name}.pt"

def fixtureNames : Array String := #["x", "sin", "cos", "expected"]

def fixturesPresent : IO Bool := do
  let mut ok := true
  for name in fixtureNames do
    ok := ok && (← (fixturePath name).pathExists)
  pure ok

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate rotary fixtures."

  IO.FS.createDirAll fixtureDir

  let device := Device.CUDA 0
  let x ← torch.rand #[64, 64] false device

  let (cosCpu, sinCpu) := torch.rotary.computeFreqsPure (64 : UInt64) (64 : UInt64) 10000.0
  let cos : T #[64, 32] := cosCpu.to device
  let sin : T #[64, 32] := sinCpu.to device

  let x4 : T #[1, 64, 1, 64] := torch.reshape x #[1, 64, 1, 64]
  let expected4 := torch.rotary.applyRotaryEmb x4 cos sin
  let expected : T #[64, 64] := torch.reshape expected4 #[64, 64]

  torch.data.saveTensor x (fixturePath "x").toString
  torch.data.saveTensor sin (fixturePath "sin").toString
  torch.data.saveTensor cos (fixturePath "cos").toString
  torch.data.saveTensor expected (fixturePath "expected").toString

  let xMean := torch.nn.item (torch.nn.meanAll x)
  let eMean := torch.nn.item (torch.nn.meanAll expected)
  IO.println s!"Generated rotary fixtures in {fixtureDir} xMean={xMean} expectedMean={eMean}"

def runOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  if !(← fixturesPresent) then
    generateFixtures

  let x ← torch.data.loadTensor #[64, 64] (fixturePath "x").toString
  let sin ← torch.data.loadTensor #[64, 32] (fixturePath "sin").toString
  let cos ← torch.data.loadTensor #[64, 32] (fixturePath "cos").toString
  let expected ← torch.data.loadTensor #[64, 64] (fixturePath "expected").toString

  let output := torch.zeros_like x

  -- grid=(1,1,1), block=(128,1,1), sharedMem=0, stream=0 (default)
  rotaryFwd.launch x sin cos output 64 64 1 1 1 128 1 1 0 0

  let ok := torch.allclose expected output 1e-4 1e-4
  let mae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (output - expected)))
  let outMean := torch.nn.item (torch.nn.meanAll output)
  let expMean := torch.nn.item (torch.nn.meanAll expected)
  IO.println s!"rotary allclose={ok} mae={mae} outMean={outMean} expectedMean={expMean}"
  pure ok

unsafe def main (args : List String) : IO UInt32 := do
  let regen := args.contains "--regen"
  let genOnly := args.contains "--gen-only"

  if regen then
    generateFixtures
  else if !(← fixturesPresent) then
    generateFixtures

  if genOnly then
    return 0

  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU

unsafe def main : List String → IO UInt32 := Examples.GPU.main
