/- End-to-end FlashAttention3 validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.FlashAttn3
import Examples.GPU.FixtureRunner

namespace Examples.GPU

open torch
open Tyr.GPU.Kernels.FlashAttn3

def fa3FixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/flashattn3_256x64"⟩
  names := #["q", "k", "v", "expected_o"]
}

def fa3FixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fa3FixtureSpec name

def generateFa3Fixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate flash attention fixtures."

  IO.FS.createDirAll fa3FixtureSpec.dir
  let device := Device.CUDA 0

  let qf ← torch.randn #[1, 1, 256, 64] false device
  let kf ← torch.randn #[1, 1, 256, 64] false device
  let vf ← torch.randn #[1, 1, 256, 64] false device

  let q := torch.toBFloat16' qf
  let k := torch.toBFloat16' kf
  let v := torch.toBFloat16' vf
  -- Quantize inputs to bf16, then compute references in float32.
  let q32 := torch.toFloat' q
  let k32 := torch.toFloat' k
  let v32 := torch.toFloat' v

  let kT : T #[1, 1, 64, 256] := torch.nn.transpose k32 2 3
  let scores : T #[1, 1, 256, 256] := torch.nn.bmm4d q32 kT
  let scaled : T #[1, 1, 256, 256] := scores / 8.0
  let probs : T #[1, 1, 256, 256] := torch.nn.softmax_dim scaled (-1)
  let expectedOut32 : T #[1, 1, 256, 64] := torch.nn.bmm4d probs v32
  let expectedOut := torch.toBFloat16' expectedOut32

  torch.data.saveTensor q (fa3FixtureFile "q").toString
  torch.data.saveTensor k (fa3FixtureFile "k").toString
  torch.data.saveTensor v (fa3FixtureFile "v").toString
  torch.data.saveTensor expectedOut (fa3FixtureFile "expected_o").toString

  let outMean := torch.nn.item (torch.nn.meanAll expectedOut)
  IO.println s!"Generated FA3 fixtures in {fa3FixtureSpec.dir} outMean={outMean}"

def runFa3Once : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

  if !(← fixturesPresent fa3FixtureSpec) then
    generateFa3Fixtures

  let q ← torch.data.loadTensor #[1, 1, 256, 64] (fa3FixtureFile "q").toString
  let k ← torch.data.loadTensor #[1, 1, 256, 64] (fa3FixtureFile "k").toString
  let v ← torch.data.loadTensor #[1, 1, 256, 64] (fa3FixtureFile "v").toString
  let expectedOut ← torch.data.loadTensor #[1, 1, 256, 64] (fa3FixtureFile "expected_o").toString

  let outFwd := torch.zeros_like q
  let lseOut := torch.zeros #[1, 1, 256] false (Device.CUDA 0)
  let stream ← torch.cuda_current_stream

  -- FA3 kernel expect seqLenQ, seqLenK, headDim as KVal
  -- flashAttn3Fwd (Q_ptr K_ptr V_ptr O_ptr L_ptr seqLenQ seqLenK headDim)
  -- The grid size needs to match CTAs. blockM=64, so 256/64 = 4 CTAs.
  flashAttn3Fwd.launch q k v outFwd lseOut 256 256 64 1 4 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let outOk := torch.allclose expectedOut outFwd 3e-2 3e-2
  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (outFwd - expectedOut)))

  IO.println s!"flashattn3 fwd_ok={outOk} out_mae={outMae}"
  pure outOk

def main (args : List String) : IO UInt32 := do
  runWithFixtures args fa3FixtureSpec generateFa3Fixtures runFa3Once

end Examples.GPU

def main : List String → IO UInt32 := Examples.GPU.main
