import Tyr.Model.FLM
import LeanTest

open torch
open torch.flm

private def tinyTokens : T #[(2 : UInt64), 3] :=
  reshape (data.fromInt64Array #[0, 1, 2, 3, 4, 0]) #[(2 : UInt64), 3]

private def identityDenoiser : FlowDenoiser 3 5 Unit :=
  { forward := fun {_batch} _ x _ _ => pure x }

@[test]
def testFLMOneHotAndCorruptAtDataTime : IO Unit := do
  let tokens := tinyTokens
  let target := oneHot (vocab := 5) tokens
  let targetSum := nn.item (nn.sumAll target)
  LeanTest.assertTrue (Float.abs (targetSum - 6.0) < 1.0e-5)
    "oneHot should put one unit of mass on each token"

  let tau := full #[(2 : UInt64)] 1.0
  let (xT, target') ← corruptContinuous (vocab := 5) tokens tau
  LeanTest.assertTrue (allclose target target' 1.0e-6 1.0e-6)
    "corruptContinuous should return the same dense target"
  LeanTest.assertTrue (allclose xT target 1.0e-6 1.0e-6)
    "at t=1 corruption should equal one-hot data"

@[test]
def testFLMLossFinite : IO Unit := do
  let cfg : FlowConfig := {}
  let tau := full #[(2 : UInt64)] 0.5
  let loss ← flowLossGivenTau cfg TimeMap.identity identityDenoiser () tinyTokens tau
  let meanLoss := nn.item (nn.meanAll loss)
  LeanTest.assertTrue (not meanLoss.isNaN) "FLM loss should be finite"
  LeanTest.assertTrue (meanLoss > 0.0) "FLM loss should be positive for an untrained denoiser"

@[test]
def testFMLMPsdLossFinite : IO Unit := do
  let cfg : FlowConfig := {}
  let tauS := full #[(2 : UInt64)] 0.1
  let tauU := full #[(2 : UInt64)] 0.5
  let tauT := full #[(2 : UInt64)] 0.9
  let loss ← psdLossGivenTau cfg TimeMap.identity identityDenoiser () tinyTokens tauS tauU tauT
  let meanLoss := nn.item (nn.meanAll loss)
  LeanTest.assertTrue (not meanLoss.isNaN) "PSD loss should be finite"
  LeanTest.assertTrue (meanLoss > 0.0) "PSD loss should be positive for an untrained denoiser"

@[test]
def testFLMGenerateFiniteTokens : IO Unit := do
  let cfg : FlowConfig := {}
  let out ← generateFLM (seq := 3) (vocab := 5) (batch := 2)
    cfg TimeMap.identity identityDenoiser () 2
  let total := nn.item (nn.sumAll (toFloat' out))
  LeanTest.assertTrue (not total.isNaN) "generated token ids should be numeric"
