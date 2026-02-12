/- End-to-end FlashAttention forward validation against PyTorch SDPA fixtures. -/
import Tyr.Torch
import Tyr.GPU.Kernels.ThunderKittensFlashAttn

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels

def fixtureDir : System.FilePath := ⟨"data/gpu_fixtures/flashattn64x64"⟩

def fixturePath (name : String) : System.FilePath :=
  fixtureDir / s!"{name}.pt"

def fixtureNames : Array String := #["q", "k", "v", "expected_o"]

def fixturesPresent : IO Bool := do
  let mut ok := true
  for name in fixtureNames do
    ok := ok && (← (fixturePath name).pathExists)
  pure ok

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate flash attention fixtures."

  IO.FS.createDirAll fixtureDir
  let device := Device.CUDA 0

  let qf ← torch.rand #[1, 1, 64, 64] false device
  let kf ← torch.rand #[1, 1, 64, 64] false device
  let vf ← torch.rand #[1, 1, 64, 64] false device

  let q := torch.toBFloat16' qf
  let k := torch.toBFloat16' kf
  let v := torch.toBFloat16' vf
  let expected := torch.nn.scaled_dot_product_attention q k v 0.0 true

  torch.data.saveTensor q (fixturePath "q").toString
  torch.data.saveTensor k (fixturePath "k").toString
  torch.data.saveTensor v (fixturePath "v").toString
  torch.data.saveTensor expected (fixturePath "expected_o").toString

  let expMean := torch.nn.item (torch.nn.meanAll expected)
  IO.println s!"Generated flash attention fixtures in {fixtureDir} expectedMean={expMean}"

def runOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  if !(← fixturesPresent) then
    generateFixtures

  let q ← torch.data.loadTensor #[1, 1, 64, 64] (fixturePath "q").toString
  let k ← torch.data.loadTensor #[1, 1, 64, 64] (fixturePath "k").toString
  let v ← torch.data.loadTensor #[1, 1, 64, 64] (fixturePath "v").toString
  let expected ← torch.data.loadTensor #[1, 1, 64, 64] (fixturePath "expected_o").toString

  let out := torch.zeros_like q

  -- grid=(1,1,1), block=(128,1,1), sharedMem=0, stream=0 (default)
  tkFlashAttnFwd.launch q k v out 64 64 1 1 1 128 1 1 0 0

  let ok := torch.allclose expected out 2e-2 2e-2
  let mae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (out - expected)))
  let maxErr := torch.nn.item (torch.nn.maxAll (torch.nn.abs (out - expected)))
  IO.println s!"flashattn allclose={ok} mae={mae} max_err={maxErr}"
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
