/-
  Tests/RunLagunaNvFp4.lean

  Validates NVFP4 dequantization (Tyr.Model.Laguna.NvFp4) against PyTorch
  ground truth in Tests/fixtures/laguna/nvfp4.safetensors, on CPU and CUDA.

  This also exercises U8 and F8_E4M3 SafeTensors FFI loading, which the real
  Laguna-S-2.1-NVFP4 checkpoint depends on.

  Ground truth generator: Tests/fixtures/laguna/gen_nvfp4_fixtures.py
-/
import Tyr.Torch
import Tyr.Model.Laguna.NvFp4

open torch
open torch.laguna

private def check (cond : Bool) (msg : String) : IO Unit := do
  if cond then
    IO.println s!"PASS: {msg}"
  else
    throw (IO.userError s!"FAIL: {msg}")

/-- Max abs error and max rel error (denominator clamped by 1e-6) between
    two tensors, computed in FP32. -/
private def maxErrors (got expected : T #[]) : IO (Float × Float) := do
  let ad : T #[] := nn.abs (sub (toFloat' got) (toFloat' expected))
  let maxAbs := nn.item (nn.maxAll ad)
  let denom : T #[] := add_scalar (nn.abs (toFloat' expected)) 1e-6
  let maxRel := nn.item (nn.maxAll (nn.div ad denom))
  pure (maxAbs, maxRel)

private def deviceLabel : Device → String
  | .CPU => "cpu"
  | .CUDA i => s!"cuda:{i}"
  | .MPS => "mps"

private def findFixture : IO String := do
  let candidates := #[
    "Tests/fixtures/laguna/nvfp4.safetensors",
    "../Tests/fixtures/laguna/nvfp4.safetensors"
  ]
  for p in candidates do
    if ← System.FilePath.pathExists p then
      return p
  throw (IO.userError "nvfp4.safetensors fixture not found (run from repo root)")

/-- Load the four tensors of one case, run dequant on `device`, and report. -/
private def runMatrixCase (path : String) (device : Device) : IO Unit := do
  let packed ← torch.safetensors.loadTensor path "m_packed" #[64, 64]
  let scales ← torch.safetensors.loadTensor path "m_scales" #[64, 8]
  let globalS ← torch.safetensors.loadTensor path "m_global" #[1]
  let expected ← torch.safetensors.loadTensor path "m_expected" #[64, 128]
  if device == Device.CPU then
    check (packed.dtype == .UInt8) s!"m_packed loads as UInt8 (got {packed.dtype})"
    check (globalS.dtype == .Float32) s!"m_global loads as Float32 (got {globalS.dtype})"
    check (expected.dtype == .BFloat16) s!"m_expected loads as BFloat16 (got {expected.dtype})"
    -- NOTE: `lean_torch_get_dtype` has no float8 case, so an F8_E4M3 tensor
    -- reports `Unknown(Unknown)` via `T.dtype`. The numerical comparison below
    -- is the actual validation that F8_E4M3 loading works.
    IO.println s!"INFO: m_scales dtypeStr = {scales.dtypeStr} (F8_E4M3 expected; FFI has no float8 dtype string)"
  let got ← nvfp4.dequantMatrix (packed.to device) (scales.to device) (globalS.to device) 64 128
  let (maxAbs, maxRel) ← maxErrors got (expected.to device)
  IO.println s!"  dequantMatrix [{deviceLabel device}]: maxAbs={maxAbs} maxRel={maxRel}"
  check (maxAbs <= 1e-3) s!"dequantMatrix [{deviceLabel device}] maxAbs={maxAbs} <= 1e-3"

private def runBankCase (path : String) (device : Device) : IO Unit := do
  let packed ← torch.safetensors.loadTensor path "b_packed" #[4, 32, 32]
  let scales ← torch.safetensors.loadTensor path "b_scales" #[4, 32, 4]
  let globalS ← torch.safetensors.loadTensor path "b_global" #[4]
  let expected ← torch.safetensors.loadTensor path "b_expected" #[4, 32, 64]
  let got ← nvfp4.dequantBank (packed.to device) (scales.to device) (globalS.to device) 4 32 64
  let (maxAbs, maxRel) ← maxErrors got (expected.to device)
  IO.println s!"  dequantBank [{deviceLabel device}]: maxAbs={maxAbs} maxRel={maxRel}"
  check (maxAbs <= 1e-3) s!"dequantBank [{deviceLabel device}] maxAbs={maxAbs} <= 1e-3"

def main : IO Unit := do
  let path ← findFixture
  IO.println s!"Using fixture: {path}"

  IO.println "-- dequantMatrix / dequantBank on CPU"
  runMatrixCase path Device.CPU
  runBankCase path Device.CPU

  if ← torch.cuda_is_available then
    IO.println "-- dequantMatrix / dequantBank on CUDA"
    let device := Device.CUDA 0
    runMatrixCase path device
    runBankCase path device
    torch.cuda_synchronize
  else
    IO.println "CUDA not available; skipped CUDA cases."

  IO.println "All Laguna NVFP4 dequantization tests passed."
