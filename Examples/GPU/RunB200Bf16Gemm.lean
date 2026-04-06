/- End-to-end Blackwell/B200 BF16 GEMM validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.Bf16Gemm
import Examples.GPU.Parity
import Examples.GPU.FixtureRunner

namespace Examples.GPU.RunB200Bf16Gemm

open torch
open Tyr.GPU.Kernels.Bf16Gemm

def suiteName : String := "b200_bf16_gemm"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/b200_bf16_gemm_256x256x64"⟩
  names := #["a", "b", "expected_c"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate b200 bf16 GEMM fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device := Device.CUDA 0

  let aFloat ← torch.randn #[256, 64] false device
  let bFloat ← torch.randn #[64, 256] false device
  let a := torch.toBFloat16' aFloat
  let b := torch.toBFloat16' bFloat
  let expectedFloat : T #[256, 256] := torch.nn.matmul2d (torch.toFloat' a) (torch.toFloat' b)
  let expected := torch.toBFloat16' expectedFloat

  torch.data.saveTensor a (fixtureFile "a").toString
  torch.data.saveTensor b (fixtureFile "b").toString
  torch.data.saveTensor expected (fixtureFile "expected_c").toString

  let aMean := torch.nn.item (torch.nn.meanAll aFloat)
  let cMean := torch.nn.item (torch.nn.meanAll expectedFloat)
  IO.println s!"Generated b200 bf16 GEMM fixtures in {fixtureSpec.dir} aMean={aMean} expectedMean={cMean}"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  if !(← isBlackwellFamily) then
    IO.println s!"[skip] {suiteName}: requires TYR_GPU_FAMILY=BLACKWELL"
    return true

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let a ← torch.data.loadTensor #[256, 64] (fixtureFile "a").toString
  let b ← torch.data.loadTensor #[64, 256] (fixtureFile "b").toString
  let expected ← torch.data.loadTensor #[256, 256] (fixtureFile "expected_c").toString

  let output := torch.zeros_like expected
  let stream ← torch.cuda_current_stream

  -- One CTA covers the full 256x256 tile for this focused parity case.
  tkB200Bf16GemmFwd.launch a b output 256 256 64 1 1 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let check := compareTensors "b200_bf16_gemm.output" expected output 5e-2 5e-2
  logTensorCheck check
  pure check.ok

def main (args : List String) : IO UInt32 := do
  runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunB200Bf16Gemm
