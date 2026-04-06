import LeanTest
import Tyr.Torch
import Examples.GPU.Parity
import Examples.GPU.FixtureRunner
import Examples.GPU.RunCopy
import Examples.GPU.RunRotary
import Examples.GPU.RunLayerNorm
import Examples.GPU.RunFlashAttn
import Examples.GPU.RunMhaH100

namespace Tests.GPUE2E

open LeanTest
open torch

private def runGpuBoolTestIf
    (label : String)
    (enabled : IO Bool)
    (reason : String)
    (action : IO Bool)
    : IO Unit := do
  if !(← torch.cuda_is_available) then
    IO.println s!"[skip] {label}: CUDA unavailable"
  else if !(← enabled) then
    IO.println s!"[skip] {label}: {reason}"
  else
    let ok ← action
    LeanTest.assertTrue ok label

private def runGpuBoolTest (label : String) (action : IO Bool) : IO Unit := do
  runGpuBoolTestIf label (pure true) "" action

private def ensureFixtures
    (spec : Examples.GPU.FixtureSpec)
    (generate : IO Unit)
    : IO Unit := do
  if !(← Examples.GPU.fixturesPresent spec) then
    generate

private def runVendoredSuiteTest
    (label : String)
    (suite : String)
    (spec : Examples.GPU.FixtureSpec)
    (generate : IO Unit)
    : IO Unit :=
  runGpuBoolTest label do
    ensureFixtures spec generate
    Examples.GPU.runVendoredReferenceIfConfigured suite spec.dir

@[test]
def testCopyTorchParity : IO Unit :=
  runGpuBoolTest "copy_tyr_vs_torch" Examples.GPU.RunCopy.runOnce

@[test]
def testRotaryTorchParity : IO Unit :=
  runGpuBoolTest "rotary_tyr_vs_torch" Examples.GPU.RunRotary.runOnce

@[test]
def testLayerNormTorchParity : IO Unit :=
  runGpuBoolTestIf
    "layernorm_tyr_vs_torch"
    (do pure !(← Examples.GPU.gpuTargetIsAny #["GB10", "B200", "B300"]))
    "current kernel is Hopper-only; no validated Blackwell-family implementation exists yet"
    Examples.GPU.RunLayerNorm.runOnce

@[test]
def testFlashAttnTorchParity : IO Unit :=
  runGpuBoolTest "flashattn_tyr_vs_torch" Examples.GPU.RunFlashAttn.runOnce

@[test]
def testMhaH100TorchParity : IO Unit :=
  runGpuBoolTest "mha_h100_tyr_vs_torch" Examples.GPU.RunMhaH100.runOnce

@[test]
def testFlashAttnVendoredParity : IO Unit :=
  runVendoredSuiteTest
    "flashattn_vendor_vs_torch"
    Examples.GPU.RunFlashAttn.suiteName
    Examples.GPU.RunFlashAttn.fixtureSpec
    Examples.GPU.RunFlashAttn.generateFixtures

@[test]
def testRotaryVendoredParity : IO Unit :=
  runVendoredSuiteTest
    "rotary_vendor_vs_torch"
    Examples.GPU.RunRotary.suiteName
    Examples.GPU.RunRotary.fixtureSpec
    Examples.GPU.RunRotary.generateFixtures

@[test]
def testLayerNormVendoredParity : IO Unit :=
  runVendoredSuiteTest
    "layernorm_vendor_vs_torch"
    Examples.GPU.RunLayerNorm.suiteName
    Examples.GPU.RunLayerNorm.fixtureSpec
    Examples.GPU.RunLayerNorm.generateFixtures

@[test]
def testMhaH100VendoredParity : IO Unit :=
  runVendoredSuiteTest
    "mha_h100_vendor_vs_torch"
    Examples.GPU.RunMhaH100.suiteName
    Examples.GPU.RunMhaH100.fixtureSpec
    Examples.GPU.RunMhaH100.generateFixtures

end Tests.GPUE2E
