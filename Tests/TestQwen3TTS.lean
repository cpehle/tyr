import Tyr
import Tyr.Model.Qwen3TTS
import LeanTest

open torch
open torch.qwen3tts

private def tinyCfg : Qwen3TTSConfig :=
  { talkerConfig := {
      codePredictorConfig := {
        vocabSize := 64
        hiddenSize := 32
        intermediateSize := 64
        numHiddenLayers := 1
        numAttentionHeads := 4
        numKeyValueHeads := 2
        headDim := 8
      }
      vocabSize := 128
      hiddenSize := 32
      intermediateSize := 64
      numHiddenLayers := 2
      numAttentionHeads := 4
      numKeyValueHeads := 2
      headDim := 8
      numCodeGroups := 4
      textHiddenSize := 32
      textVocabSize := 128
      codecPadId := 0
      codecBosId := 3
      codecEosTokenId := 7
      codecThinkId := 8
      codecNoThinkId := 9
      codecThinkBosId := 10
      codecThinkEosId := 11
    }
    speakerEncoderConfig := {
      melDim := 16
      encDim := 32
      sampleRate := 24000
    }
    ttsModelType := "base"
  }

@[test]
def testQwen3TTSInitAndCapabilities : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let langs := model.getSupportedLanguages
  LeanTest.assertTrue (langs.size >= 1) "Supported languages should include at least auto"

@[test]
def testQwen3TTSGenerateFromText : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg

  let batch : UInt64 := 2
  let textSeq : UInt64 := 6
  let maxFrames : UInt64 := 5

  let textIds ← randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[batch, textSeq]
  let out ← model.generateFromText textIds maxFrames

  LeanTest.assertEqual out.lengths.size batch.toNat "lengths should have one value per batch row"

  -- Verify padded output shape by materializing first codebook and flattening.
  let firstBook3 : T #[batch, maxFrames, 1] := data.slice out.codes 2 0 1
  let firstBook2 : T #[batch, maxFrames] := reshape firstBook3 #[batch, maxFrames]
  let flat : T #[batch * maxFrames] := reshape firstBook2 #[batch * maxFrames]
  let toks ← data.tensorToUInt64Array flat
  LeanTest.assertEqual toks.size (batch * maxFrames).toNat "flattened token count should match batch*frames"

  for len in out.lengths do
    LeanTest.assertTrue (len >= 1 && len <= maxFrames) "each generated length must be within [1, maxFrames]"

@[test]
def testQwen3TTSSpeakerEmbedding : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let batch : UInt64 := 2
  let frames : UInt64 := 12
  let mel ← randn #[batch, frames, cfg.speakerEncoderConfig.melDim]
  let emb ← model.extractSpeakerEmbedding mel
  let s := nn.item (nn.sumAll emb)
  LeanTest.assertTrue (Float.isFinite s) "speaker embedding sum should be finite"

@[test]
def testQwen3TTSStreamingApi : IO Unit := do
  let cfg := tinyCfg
  let model ← Qwen3TTSForConditionalGeneration.init cfg
  let textIds : T #[1, 5] ← randint 0 cfg.talkerConfig.textVocabSize.toInt64 #[1, 5]
  let talkerInputs : T #[1, 6, cfg.talkerConfig.hiddenSize] := model.buildTalkerInputsFromText textIds

  let callbacksSeenRef ← IO.mkRef (0 : Nat)
  let callbacks : Qwen3TTSForConditionalGeneration.StreamingCallbacks := {
    onCodeFrame := fun _ row => do
      LeanTest.assertEqual row.size cfg.talkerConfig.numCodeGroups.toNat "stream row width should equal numCodeGroups"
      callbacksSeenRef.modify (fun n => n + 1)
  }
  let opts : Qwen3TTSForConditionalGeneration.StreamingOptions cfg := {
    maxFrames := 4
    minNewTokens := 1
    emitEosFrame := true
  }
  let out ← model.streamFromTalkerInputs talkerInputs opts none callbacks
  LeanTest.assertEqual out.lengths.size 1 "stream lengths should have one value for batch=1"
  LeanTest.assertTrue (out.codeRows.size <= opts.maxFrames.toNat) "streamed rows should not exceed maxFrames"
  let callbacksSeen ← callbacksSeenRef.get
  LeanTest.assertEqual callbacksSeen out.codeRows.size "callback should fire once per emitted stream row"
