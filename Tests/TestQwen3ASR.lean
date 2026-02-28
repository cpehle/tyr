import Tyr
import Tyr.Model.Qwen3ASR
import LeanTest

open torch
open torch.qwen3asr

private def tinyCfg : Qwen3ASRConfig :=
  { thinkerConfig := {
      audioConfig := {
        numMelBins := 8
        encoderLayers := 2
        encoderAttentionHeads := 2
        encoderFfnDim := 32
        dModel := 16
        outputDim := 16
        downsampleHiddenSize := 4
      }
      textConfig := {
        vocabSize := 96
        hiddenSize := 16
        intermediateSize := 32
        numHiddenLayers := 2
        numAttentionHeads := 4
        numKeyValueHeads := 2
        headDim := 4
        maxPositionEmbeddings := 1024
        ropeTheta := 10000.0
      }
      audioTokenId := 42
    }
    supportLanguages := #["English", "Chinese"]
  }

private def tinyForcedAlignerCfg : Qwen3ASRConfig :=
  { thinkerConfig := {
      audioConfig := {
        numMelBins := 8
        encoderLayers := 2
        encoderAttentionHeads := 2
        encoderFfnDim := 32
        dModel := 16
        outputDim := 16
        downsampleHiddenSize := 4
      }
      textConfig := {
        vocabSize := 96
        hiddenSize := 16
        intermediateSize := 32
        numHiddenLayers := 2
        numAttentionHeads := 4
        numKeyValueHeads := 2
        headDim := 4
        maxPositionEmbeddings := 1024
        ropeTheta := 10000.0
      }
      audioTokenId := 42
      modelType := "qwen3_forced_aligner_thinker"
      classifyNum := 7
    }
    supportLanguages := #["English"]
  }

private def tinyPreprocessorCfg (melBins frames : UInt64) : PreprocessorConfig := {
  featureExtractorType := "WhisperFeatureExtractor"
  featureSize := melBins
  samplingRate := 16000
  hopLength := 160
  chunkLength := 1
  nFft := 400
  nSamples := frames * 160
  nbMaxFrames := frames
  paddingSide := "right"
  paddingValue := 0.0
  returnAttentionMask := true
  doNormalize := false
  dither := 0.0
}

@[test]
def testQwen3ASRInitAndLanguages : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  LeanTest.assertEqual (model.getSupportedLanguages).size 2 "expected two configured languages"

@[test]
def testQwen3ASRAudioForward : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 2
  let frames : UInt64 := 32
  let mel ← randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let audio := model.thinker.encodeAudio mel
  let s := nn.item (nn.sumAll audio)
  LeanTest.assertTrue (Float.isFinite s) "audio encoder output sum should be finite"

@[test]
def testQwen3ASRForwardFromMel : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg

  let batch : UInt64 := 2
  let frames : UInt64 := 32
  let textSeq : UInt64 := 6
  let outSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq

  let mel ← randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let textIds ← randint 0 cfg.thinkerConfig.textConfig.vocabSize.toInt64 #[batch, textSeq]
  let logits := model.forwardFromMel mel textIds

  let s := nn.item (nn.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "logit sum should be finite"

  let (_vals, idxs) := max_dim_3d logits 2
  let flat : T #[batch * outSeq] := reshape idxs #[batch * outSeq]
  let toks ← data.tensorToUInt64Array flat
  LeanTest.assertEqual toks.size (batch * outSeq).toNat "flattened argmax token count should match output length"

@[test]
def testQwen3ASRFaithfulForwardPlaceholderScatter : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg

  let batch : UInt64 := 1
  let frames : UInt64 := 32
  let audioSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames
  let seq : UInt64 := audioSeq + 4

  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let idsVals : Array Int64 := #[aTok, aTok, aTok, aTok, 1, 2, 3, 4]
  let inputIds : T #[batch, seq] := reshape (data.fromInt64Array idsVals) #[batch, seq]

  let mel ← randn #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]
  let featureMask : T #[batch, frames] := full_int #[batch, frames] 1

  let logits ← model.forward inputIds (some mel) (some featureMask) none
  let s := nn.item (nn.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "faithful forward logits should be finite"

@[test]
def testQwen3ASRForwardFromOutputWav : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg

  let wavPath := "output/lean_decode_test.wav"
  let wavExists ← data.fileExists wavPath
  LeanTest.assertTrue wavExists s!"expected tracked test WAV at {wavPath}"

  let batch : UInt64 := 1
  let frames : UInt64 := 64
  let textSeq : UInt64 := 5
  let outSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq

  let frontendCfg := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins frames
  let frontendOut ← wavToWhisperFeatures frontendCfg wavPath
  let mel := frontendOut.inputFeatures
  let textIds ← randint 0 cfg.thinkerConfig.textConfig.vocabSize.toInt64 #[batch, textSeq]
  let logits := model.forwardFromMel mel textIds

  let s := nn.item (nn.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "logit sum from WAV-derived mel should be finite"

  let (_vals, idxs) := max_dim_3d logits 2
  let flat : T #[batch * outSeq] := reshape idxs #[batch * outSeq]
  let toks ← data.tensorToUInt64Array flat
  LeanTest.assertEqual toks.size (batch * outSeq).toNat "flattened argmax token count should match output length"

@[test]
def testQwen3ASRPreprocessorConfigLoad : IO Unit := do
  let path := "/tmp/qwen3asr_preprocessor_config_test.json"
  let json :=
    "{\n"
    ++ "  \"feature_extractor_type\": \"WhisperFeatureExtractor\",\n"
    ++ "  \"feature_size\": 8,\n"
    ++ "  \"sampling_rate\": 16000,\n"
    ++ "  \"hop_length\": 160,\n"
    ++ "  \"chunk_length\": 2,\n"
    ++ "  \"n_fft\": 400,\n"
    ++ "  \"padding_side\": \"right\",\n"
    ++ "  \"padding_value\": 0.0,\n"
    ++ "  \"return_attention_mask\": true,\n"
    ++ "  \"do_normalize\": false,\n"
    ++ "  \"dither\": 0.0\n"
    ++ "}\n"
  IO.FS.writeFile path json
  let cfg ← PreprocessorConfig.loadFromFile path
  LeanTest.assertEqual cfg.featureSize 8 "feature_size should be loaded"
  LeanTest.assertEqual cfg.samplingRate 16000 "sampling_rate should be loaded"
  LeanTest.assertEqual cfg.nFft 400 "n_fft should be loaded"
  LeanTest.assertEqual (PreprocessorConfig.expectedSampleCount cfg) 32000
    "n_samples should be derived from chunk_length * sampling_rate when missing"
  LeanTest.assertEqual (PreprocessorConfig.expectedFrames cfg) 200
    "nb_max_frames should be derived from n_samples / hop_length when missing"

@[test]
def testQwen3ASRProcessorReplaceSpecialTokens : IO Unit := do
  let p : Qwen3ASRProcessor := { audioToken := "<|AUDIO|>" }
  let text := #["x<|AUDIO|>y<|AUDIO|>z", "pre<|AUDIO|>post"]
  let lens := #[2, 3, 1]
  let out ←
    match p.replaceMultimodalSpecialTokens text lens with
    | .ok xs => pure xs
    | .error e => throw <| IO.userError s!"replaceMultimodalSpecialTokens failed: {e}"

  LeanTest.assertEqual (out.getD 0 "") "x<|AUDIO|><|AUDIO|>y<|AUDIO|><|AUDIO|><|AUDIO|>z"
    "first sample token replacement should match reference semantics"
  LeanTest.assertEqual (out.getD 1 "") "pre<|AUDIO|>post"
    "second sample should consume one audio length and keep token spelling"

@[test]
def testQwen3ASRProcessorChunkedIndex : IO Unit := do
  let idx := Qwen3ASRProcessor.getChunkedIndex #[0, 10, 999, 1000, 1300, 2200] 1000
  LeanTest.assertEqual idx #[(0, 3), (3, 5), (5, 6)]
    "chunk boundaries should follow token-range thresholds"

@[test]
def testQwen3ASRFrontendWavWhisperFeatures : IO Unit := do
  let wavPath := "output/lean_decode_test.wav"
  let wavExists ← data.fileExists wavPath
  LeanTest.assertTrue wavExists s!"expected tracked test WAV at {wavPath}"

  let cfg := tinyPreprocessorCfg 8 64
  let out ← wavToWhisperFeatures cfg wavPath
  let mel := out.inputFeatures
  let fmask := out.featureAttentionMask

  LeanTest.assertEqual mel.runtimeShape #[1, 8, 64] "frontend mel runtime shape should match config"
  LeanTest.assertEqual fmask.runtimeShape #[1, 64] "feature mask runtime shape should match config"

  let s := nn.item (nn.sumAll mel)
  LeanTest.assertTrue (Float.isFinite s) "Whisper frontend mel sum should be finite"

  let flatMask : T #[64] := reshape fmask #[64]
  let maskVals ← data.tensorToUInt64Array (data.toLong flatMask)
  LeanTest.assertEqual maskVals.size 64 "feature mask length should equal configured frames"
  LeanTest.assertTrue (maskVals.all (fun v => v == 0 || v == 1))
    "feature mask values should be binary"

  -- Reference values generated from transformers WhisperFeatureExtractor
  -- with matching config:
  -- feature_size=8, sampling_rate=16000, hop_length=160, chunk_length=1,
  -- n_fft=400, padding='max_length', truncation=true, max_length=10240.
  let refFirst16 : Array Float := #[
    -0.21674562, -0.29741502, -0.31741607, -0.36881697,
    -0.40295756, -0.42537963, -0.42546642, -0.53501916,
    -0.5185493, -0.57839763, -0.5599686, -0.51503634,
    -0.48306525, -0.5830481, -0.49611127, -0.54732037
  ]
  let melFlat : T #[8 * 64] := reshape mel #[8 * 64]
  let melVals ← data.tensorToFloatArray' (reshape melFlat #[])
  let mut maxErr : Float := 0.0
  for i in [:refFirst16.size] do
    let err := Float.abs (melVals.getD i 0.0 - refFirst16[i]!)
    if err > maxErr then
      maxErr := err
  LeanTest.assertTrue (maxErr < 0.01)
    s!"frontend mel should match Whisper reference on first 16 values (max_err={maxErr})"

@[test]
def testQwen3ASRForwardForcedAlignerHead : IO Unit := do
  let cfg := tinyForcedAlignerCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 2
  let seq : UInt64 := 5
  let textIds ← randint 0 cfg.thinkerConfig.textConfig.vocabSize.toInt64 #[batch, seq]
  let logits := model.forwardText textIds
  let s := nn.item (nn.sumAll logits)
  LeanTest.assertTrue (Float.isFinite s) "forced-aligner logits should be finite"

  let (_vals, idxs) := max_dim_3d logits 2
  let flat : T #[batch * seq] := reshape idxs #[batch * seq]
  let ids ← data.tensorToUInt64Array flat
  LeanTest.assertTrue (ids.all (fun i => i < cfg.thinkerConfig.classifyNum))
    "argmax ids should be within forced-aligner classify_num range"

@[test]
def testQwen3ASRForwardWithAuxRopeDeltas : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let batch : UInt64 := 1
  let seq : UInt64 := 6
  let ids ← randint 0 cfg.thinkerConfig.textConfig.vocabSize.toInt64 #[batch, seq]
  let attnVals : Array Int64 := #[0, 0, 1, 1, 1, 1]
  let attnMask : T #[batch, seq] := reshape (data.fromInt64Array attnVals) #[batch, seq]
  let out ← model.forwardWithAux
    ids
    (inputFeatures := (none : Option (T #[batch, cfg.thinkerConfig.audioConfig.numMelBins, 1])))
    (featureAttentionMask := (none : Option (T #[batch, 1])))
    (attentionMask := some attnMask)
  match out.ropeDeltas with
  | none =>
    LeanTest.assertTrue false "forwardWithAux should populate ropeDeltas when attention mask is provided"
  | some d =>
    let flat : T #[batch] := reshape d #[batch]
    let arr ← data.tensorToUInt64Array (data.toLong flat)
    LeanTest.assertEqual arr.size batch.toNat "ropeDeltas batch size should match"

@[test]
def testQwen3ASRGreedyGenerateFromWav : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3ASRForConditionalGeneration.init cfg
  let wavPath := "output/lean_decode_test.wav"
  let wavExists ← data.fileExists wavPath
  LeanTest.assertTrue wavExists s!"expected tracked test WAV at {wavPath}"

  let batch : UInt64 := 1
  let frames : UInt64 := 64
  let audioSeq : UInt64 := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames
  let promptSeq : UInt64 := audioSeq + 2
  let aTok : Int64 := Int64.ofNat cfg.thinkerConfig.audioTokenId.toNat
  let mut idsVals : Array Int64 := #[]
  for _ in [:audioSeq.toNat] do
    idsVals := idsVals.push aTok
  idsVals := idsVals.push 5
  idsVals := idsVals.push 9
  let inputIds : T #[batch, promptSeq] := reshape (data.fromInt64Array idsVals) #[batch, promptSeq]

  let frontendCfg := tinyPreprocessorCfg cfg.thinkerConfig.audioConfig.numMelBins frames
  let frontendOut ← wavToWhisperFeatures frontendCfg wavPath
  let mel : T #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames] := frontendOut.inputFeatures
  let fmask : T #[batch, frames] := frontendOut.featureAttentionMask

  let generated ← model.generateGreedy inputIds (some mel) (some fmask) 3 #[]
  let outSeq := generated.1
  let outIds := generated.2
  LeanTest.assertEqual outSeq (promptSeq + 3) "greedy generation should append max_new_tokens when eos set is empty"

  let flat : T #[outSeq] := reshape outIds #[outSeq]
  let toks ← data.tensorToUInt64Array flat
  LeanTest.assertEqual toks.size outSeq.toNat "generated token tensor should match reported output sequence length"
