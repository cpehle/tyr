/-
  Tests/TestModularNorm.lean

  Tests for the modular norm implementation.
-/
import LeanTest
import Tyr.Modular

open LeanTest
open Tyr.Modular

/-! ## Basic Norm Property Tests -/

@[test]
def testSpectralNormPositive : IO Unit := do
  -- Spectral norm should be non-negative
  let mat ← torch.randn #[4, 3]
  let σ := torch.linalg.spectralNorm mat
  LeanTest.assertTrue (σ ≥ 0.0) s!"Spectral norm should be non-negative, got {σ}"

@[test]
def testSpectralNormOfZero : IO Unit := do
  -- Spectral norm of zero matrix should be 0
  let mat := torch.zeros #[3, 3]
  let σ := torch.linalg.spectralNorm mat
  LeanTest.assertTrue (σ < 1e-6) s!"Spectral norm of zero should be ~0, got {σ}"

@[test]
def testNuclearNormPositive : IO Unit := do
  -- Nuclear norm should be non-negative
  let mat ← torch.randn #[4, 3]
  let n := torch.linalg.nuclearNorm mat
  LeanTest.assertTrue (n ≥ 0.0) s!"Nuclear norm should be non-negative, got {n}"

@[test]
def testSpectralLessThanNuclear : IO Unit := do
  -- Spectral norm ≤ nuclear norm (equality only for rank-1 matrices)
  let mat ← torch.randn #[4, 3]
  let σ := torch.linalg.spectralNorm mat
  let n := torch.linalg.nuclearNorm mat
  LeanTest.assertTrue (σ ≤ n + 1e-5)
    s!"Spectral {σ} should be ≤ nuclear {n}"

@[test]
def testL2NormPositive : IO Unit := do
  -- L2 norm should be non-negative
  let vec ← torch.randn #[10]
  let n := torch.linalg.l2Norm vec
  LeanTest.assertTrue (n ≥ 0.0) s!"L2 norm should be non-negative, got {n}"

@[test]
def testMaxRowNormPositive : IO Unit := do
  -- Max row norm should be non-negative
  let mat ← torch.randn #[5, 3]
  let n := torch.linalg.maxRowNorm mat
  LeanTest.assertTrue (n ≥ 0.0) s!"Max row norm should be non-negative, got {n}"

/-! ## Linear Layer NormedModule Tests -/

@[test]
def testLinearNormPositive : IO Unit := do
  -- Linear layer norm should be positive for non-zero weights
  let lin ← torch.Linear.init 10 5
  let n := NormedModule.norm lin
  LeanTest.assertTrue (n > 0.0) s!"Linear norm should be positive, got {n}"

@[test]
def testLinearNuPositive : IO Unit := do
  -- Input sensitivity should be positive
  let lin ← torch.Linear.init 10 5
  let ν := NormedModule.nu lin
  LeanTest.assertTrue (ν > 0.0) s!"Linear nu should be positive, got {ν}"

@[test]
def testLinearNormalizationReducesNorm : IO Unit := do
  -- After normalization, norm should be close to 1
  let lin ← torch.Linear.init 10 5
  let normalizedLin := NormedModule.normalize lin
  let n := NormedModule.norm normalizedLin
  -- Normalized spectral norm should be ~1 (we normalize weight to unit spectral norm)
  LeanTest.assertTrue (n < 2.0) s!"Normalized linear norm should be ~1, got {n}"

/-! ## LayerNorm NormedModule Tests -/

@[test]
def testLayerNormNormPositive : IO Unit := do
  -- LayerNorm norm should be positive (initialized with ones/zeros)
  let ln := torch.LayerNorm.init 64
  let n := NormedModule.norm ln
  LeanTest.assertTrue (n > 0.0) s!"LayerNorm norm should be positive, got {n}"

@[test]
def testLayerNormNuIsOne : IO Unit := do
  -- LayerNorm should have nu = 1 (approximately preserves scale)
  let ln := torch.LayerNorm.init 64
  let ν := NormedModule.nu ln
  LeanTest.assertTrue (ν == 1.0) s!"LayerNorm nu should be 1.0, got {ν}"

/-! ## Embedding NormedModule Tests -/

@[test]
def testEmbeddingNormPositive : IO Unit := do
  -- Embedding norm should be positive
  let emb ← Embedding.init 1000 64
  let n := NormedModule.norm emb
  LeanTest.assertTrue (n > 0.0) s!"Embedding norm should be positive, got {n}"

@[test]
def testEmbeddingDualNormPositive : IO Unit := do
  -- Embedding dual norm (sum of row norms) should be positive
  let emb ← Embedding.init 1000 64
  let d := NormedModule.dualNorm emb
  LeanTest.assertTrue (d > 0.0) s!"Embedding dual norm should be positive, got {d}"

/-! ## Composition Tests -/

@[test]
def testSequentialCompositionNorm : IO Unit := do
  -- Test that sequential composition computes correctly
  let lin1 ← torch.Linear.init 10 20
  let lin2 ← torch.Linear.init 20 5
  let seq := compose lin1 lin2
  let n := NormedModule.norm seq
  LeanTest.assertTrue (n > 0.0) s!"Sequential norm should be positive, got {n}"

@[test]
def testSequentialNuMultiplies : IO Unit := do
  -- For sequential composition, ν should multiply
  let lin1 ← torch.Linear.init 10 20
  let lin2 ← torch.Linear.init 20 5
  let ν1 := NormedModule.nu lin1
  let ν2 := NormedModule.nu lin2
  let seq := compose lin1 lin2
  let νSeq := NormedModule.nu seq
  let expected := ν1 * ν2
  let relError := Float.abs (νSeq - expected) / expected
  LeanTest.assertTrue (relError < 0.01)
    s!"Sequential nu should multiply: {νSeq} vs expected {expected}"

@[test]
def testParallelCompositionNorm : IO Unit := do
  -- Test that parallel composition uses L2 combination
  let lin1 ← torch.Linear.init 10 5
  let lin2 ← torch.Linear.init 10 5
  let n1 := NormedModule.norm lin1
  let n2 := NormedModule.norm lin2
  let par : torch.Linear 10 5 × torch.Linear 10 5 := (lin1, lin2)
  let nPar := NormedModule.norm par
  let expected := Float.sqrt (n1 * n1 + n2 * n2)
  let relError := Float.abs (nPar - expected) / expected
  LeanTest.assertTrue (relError < 0.01)
    s!"Parallel norm should be L2: {nPar} vs expected {expected}"

/-! ## Array/Vector Tests -/

@[test]
def testArrayNormPositive : IO Unit := do
  -- Array of modules should have positive norm
  let layers ← (List.range 3).mapM fun _ => torch.Linear.init 10 10
  let arr := layers.toArray
  let n := NormedModule.norm arr
  LeanTest.assertTrue (n > 0.0) s!"Array norm should be positive, got {n}"

@[test]
def testArrayNormIsL2 : IO Unit := do
  -- Array norm should be L2 combination
  let lin1 ← torch.Linear.init 10 10
  let lin2 ← torch.Linear.init 10 10
  let arr := #[lin1, lin2]
  let n1 := NormedModule.norm lin1
  let n2 := NormedModule.norm lin2
  let nArr := NormedModule.norm arr
  let expected := Float.sqrt (n1 * n1 + n2 * n2)
  let relError := Float.abs (nArr - expected) / expected
  LeanTest.assertTrue (relError < 0.01)
    s!"Array norm should be L2: {nArr} vs expected {expected}"

/-! ## Normalization Tests -/

@[test]
def testNormalizeDualReducesDualNorm : IO Unit := do
  -- Dual normalization should reduce dual norm
  let lin ← torch.Linear.init 10 5
  let dBefore := NormedModule.dualNorm lin
  let normalized := NormedModule.normalizeDual lin
  let dAfter := NormedModule.dualNorm normalized
  LeanTest.assertTrue (dAfter < dBefore + 1e-5)
    s!"Dual norm should decrease after normalization: {dAfter} vs {dBefore}"
