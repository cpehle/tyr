import Tyr
import Tyr.Model.KittenTTS
import LeanTest

open torch
open torch.kittentts

private def tinyCfg : KittenTTSConfig :=
  {
    hiddenDim := 8
    maxConvDim := 8
    maxDur := 4
    nLayer := 1
    nMels := 8
    nToken := 32
    styleDim := 4
    textEncoderKernelSize := 3
    asrResDim := 4
    decoderOutDim := 8
    plbert := {
      numHiddenLayers := 1
      numAttentionHeads := 2
      hiddenSize := 8
      intermediateSize := 16
      maxPositionEmbeddings := 32
      embeddingSize := 4
      innerGroupNum := 1
      numHiddenGroups := 1
      typeVocabSize := 2
    }
    generator := {
      resblockKernelSizes := #[3]
      upsampleRates := #[2]
      upsampleInitialChannel := 8
      resblockDilationSizes := #[#[1, 3, 5]]
      upsampleKernelSizes := #[4]
      genIstftNFft := 8
      genIstftHopSize := 2
      harmonicCount := 2
    }
  }

@[test]
def testKittenTTSInitAndForward : IO Unit := do
  let cfg := tinyCfg
  let model ← Model.init cfg
  let inputIds : T #[1, 5] ← randint 0 cfg.nToken.toInt64 #[1, 5]
  let refStyle : T #[1, KittenTTSConfig.fullStyleDim cfg] ← randn #[1, KittenTTSConfig.fullStyleDim cfg]
  let out ← model.synthesizeIds inputIds refStyle
  let debugOut ← model.debugSynthesizeIds inputIds refStyle

  LeanTest.assertEqual out.predDurations.size 5 "predicted durations should align with the input token count"
  LeanTest.assertTrue (out.alignmentFrames >= 5) "each token should contribute at least one frame"
  LeanTest.assertEqual debugOut.predDurations out.predDurations
    "debug synthesis should preserve duration predictions"
  LeanTest.assertEqual debugOut.alignmentFrames out.alignmentFrames
    "debug synthesis should preserve alignment length"
  let audioShape := out.audio.runtimeShape
  LeanTest.assertEqual (audioShape.getD 0 0) 1 "audio batch dimension should be 1"
  LeanTest.assertEqual (audioShape.getD 1 0) 1 "audio channel dimension should be 1"
  LeanTest.assertEqual
    (audioShape.getD 2 0)
    (GeneratorConfig.waveSamples cfg.generator (2 * out.alignmentFrames))
    "audio sample length should match the full iSTFT vocoder expansion"

  let audioVals ← data.tensorToFloatArray' out.audio
  let debugAudioVals ← data.tensorToFloatArray' debugOut.audio
  LeanTest.assertTrue (!audioVals.isEmpty) "audio output should contain samples"
  LeanTest.assertEqual debugAudioVals.size audioVals.size
    "debug synthesis audio should match the regular forward path sample count"
  for v in audioVals do
    LeanTest.assertTrue (Float.isFinite v) "audio values should be finite"
  for i in [:audioVals.size] do
    let dv := debugAudioVals[i]!
    let v := audioVals[i]!
    LeanTest.assertTrue (Float.abs (dv - v) <= 1e-6)
      "debug synthesis audio should stay numerically aligned with the normal forward path"
