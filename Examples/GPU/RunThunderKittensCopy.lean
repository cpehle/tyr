/-
  Examples/GPU/RunThunderKittensCopy.lean

  End-to-end GPU demo:
  - create CUDA torch tensors in Lean
  - launch generated tkCopy kernel
  - verify output matches input
-/
import Tyr.Torch
import Tyr.GPU.Kernels.ThunderKittensCopy

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels

def runOnce : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  let device := Device.CUDA 0
  let input ← torch.rand #[1, 1, 64, 64] false device
  let output := torch.zeros_like input

  -- grid=(1,1,1), block=(128,1,1), sharedMem=0, stream=0 (default)
  tkCopy.launch input output 1 1 1 128 1 1 0 0

  let ok := torch.allclose input output
  let inMean := torch.nn.item (torch.nn.meanAll input)
  let outMean := torch.nn.item (torch.nn.meanAll output)
  IO.println s!"tkCopy allclose={ok} inputMean={inMean} outputMean={outMean}"
  pure ok

def main (_ : List String) : IO UInt32 := do
  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
