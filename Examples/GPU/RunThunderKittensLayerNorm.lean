/- End-to-end ThunderKittens-style layernorm validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.ThunderKittensLayerNorm

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels

def fixtureDir : System.FilePath := ⟨"data/gpu_fixtures/layernorm64x1024"⟩

def fixturePath (name : String) : System.FilePath :=
  fixtureDir / s!"{name}.pt"

def fixtureNames : Array String :=
  #["x", "residual", "weight", "bias", "expected_out", "expected_resid"]

def fixturesPresent : IO Bool := do
  let mut ok := true
  for name in fixtureNames do
    ok := ok && (← (fixturePath name).pathExists)
  pure ok

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate layernorm fixtures."

  IO.FS.createDirAll fixtureDir
  let device := Device.CUDA 0
  let eps : Float := 1.0e-5

  let x ← torch.rand #[1, 64, 1024] false device
  let residual ← torch.rand #[1, 64, 1024] false device
  let weight ← torch.rand #[1024] false device
  let bias ← torch.rand #[1024] false device

  let expectedResid := x + residual
  let expectedOut := torch.nn.layer_norm expectedResid weight bias eps

  torch.data.saveTensor x (fixturePath "x").toString
  torch.data.saveTensor residual (fixturePath "residual").toString
  torch.data.saveTensor weight (fixturePath "weight").toString
  torch.data.saveTensor bias (fixturePath "bias").toString
  torch.data.saveTensor expectedOut (fixturePath "expected_out").toString
  torch.data.saveTensor expectedResid (fixturePath "expected_resid").toString

  let outMean := torch.nn.item (torch.nn.meanAll expectedOut)
  let residMean := torch.nn.item (torch.nn.meanAll expectedResid)
  IO.println s!"Generated layernorm fixtures in {fixtureDir} outMean={outMean} residMean={residMean}"

def runOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  if !(← fixturesPresent) then
    generateFixtures

  let x ← torch.data.loadTensor #[1, 64, 1024] (fixturePath "x").toString
  let residual ← torch.data.loadTensor #[1, 64, 1024] (fixturePath "residual").toString
  let weight ← torch.data.loadTensor #[1024] (fixturePath "weight").toString
  let bias ← torch.data.loadTensor #[1024] (fixturePath "bias").toString
  let expectedOut ← torch.data.loadTensor #[1, 64, 1024] (fixturePath "expected_out").toString
  let expectedResid ← torch.data.loadTensor #[1, 64, 1024] (fixturePath "expected_resid").toString

  let out := torch.zeros_like x
  -- Use a distinct source expression to prevent CSE from aliasing outputs.
  let outResid := torch.zeros_like residual

  -- Single-warp launch keeps numerical behavior deterministic for this kernel.
  tkLayerNorm.launch x residual weight bias out outResid 1 1 1 32 1 1 0 0

  let outOk := torch.allclose expectedOut out 5e-3 5e-3
  let residOk := torch.allclose expectedResid outResid 1e-5 1e-5

  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (out - expectedOut)))
  let residMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (outResid - expectedResid)))
  let outMax := torch.nn.item (torch.nn.maxAll (torch.nn.abs (out - expectedOut)))
  IO.println s!"layernorm out_allclose={outOk} resid_allclose={residOk} out_mae={outMae} resid_mae={residMae} out_max={outMax}"
  pure (outOk && residOk)

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
