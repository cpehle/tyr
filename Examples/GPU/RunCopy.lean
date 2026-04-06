/-
  Examples/GPU/RunCopy.lean

  End-to-end GPU demo:
  - create CUDA torch tensors in Lean
  - launch generated tkCopy kernel
  - verify output matches input
-/
import Tyr.Torch
import Tyr.GPU.Kernels.Copy
import Examples.GPU.Parity

namespace Examples.GPU.RunCopy

open torch
open Tyr.GPU.Kernels

def suiteName : String := "copy"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  seedFixtures suiteName 1001
  let device := Device.CUDA 0
  let input ← torch.rand #[1, 1, 64, 64] false device
  let output := torch.zeros_like input
  let stream ← torch.cuda_current_stream

  -- grid=(1,1,1), block=(128,1,1), sharedMem=0
  copy64x64.launch input output 1 1 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  reportTensorComparison "copy.output" input output 1e-7 1e-7

def main (_ : List String) : IO UInt32 := do
  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU.RunCopy
