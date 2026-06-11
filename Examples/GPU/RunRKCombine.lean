/-
  Examples/GPU/RunRKCombine.lean

  End-to-end GPU parity check for the fused Dopri5 stage combination:
  - create CUDA tensors for y0 and the seven stage derivatives
  - launch the fused kernel (one launch for solution + error estimate)
  - compare against the torch-composed reference combination
-/
import Tyr.Torch
import Tyr.GPU.Kernels.RKCombine
import Examples.GPU.Parity

namespace Examples.GPU.RunRKCombine

open torch
open Tyr.GPU.Kernels

def suiteName : String := "rk_combine"

private def b : Array Float :=
  #[35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0,
    -2187.0 / 6784.0, 11.0 / 84.0, 0.0]

private def bHat : Array Float :=
  #[1951.0 / 21600.0, 0.0, 22642.0 / 50085.0, 451.0 / 720.0,
    -12231.0 / 42400.0, 649.0 / 6300.0, 1.0 / 60.0]

private def combineRef (y0 : T #[1, 1, 64, 64]) (ks : Array (T #[1, 1, 64, 64]))
    (weights : Array Float) (base : Option (T #[1, 1, 64, 64])) : T #[1, 1, 64, 64] := Id.run do
  let mut acc :=
    match base with
    | some t => t
    | none => mul_scalar y0 0.0
  for i in [:ks.size] do
    let w := weights.getD i 0.0
    if w != 0.0 then
      acc := add acc (mul_scalar (ks.getD i y0) w)
  return acc

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  seedFixtures suiteName 7117
  let device := Device.CUDA 0
  let y0 ← torch.rand #[1, 1, 64, 64] false device
  let mut ks : Array (T #[1, 1, 64, 64]) := #[]
  for _ in [:7] do
    ks := ks.push (← torch.rand #[1, 1, 64, 64] false device)
  -- NB: output buffers must come from syntactically DISTINCT pure calls:
  -- `zeros_like y0` twice gets common-subexpression-eliminated into one
  -- shared tensor, and the kernel then writes both outputs into the same
  -- buffer (observed on GB10 as y1 containing the error estimate).
  let y1 := torch.zeros_like y0
  let err := torch.zeros_like (ks.getD 0 y0)
  let stream ← torch.cuda_current_stream

  let blackwell ← isBlackwellFamily
  if blackwell then
    dopri5Combine64Blackwell.launch y0
      (ks.getD 0 y0) (ks.getD 1 y0) (ks.getD 2 y0) (ks.getD 3 y0)
      (ks.getD 4 y0) (ks.getD 5 y0) (ks.getD 6 y0)
      y1 err 1 1 1 32 1 1 0 stream
  else
    dopri5Combine64.launch y0
      (ks.getD 0 y0) (ks.getD 1 y0) (ks.getD 2 y0) (ks.getD 3 y0)
      (ks.getD 4 y0) (ks.getD 5 y0) (ks.getD 6 y0)
      y1 err 1 1 1 32 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let y1Ref := combineRef y0 ks b (some y0)
  let errW := (Array.range 7).map fun i => b.getD i 0.0 - bHat.getD i 0.0
  let errRef := combineRef y0 ks errW none

  let okY ← reportTensorComparison "rk_combine.y1" y1Ref y1 1e-5 1e-5
  let okE ← reportTensorComparison "rk_combine.err" errRef err 1e-5 1e-5
  pure (okY && okE)

def main (_ : List String) : IO UInt32 := do
  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU.RunRKCombine
