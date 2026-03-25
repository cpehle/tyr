import Tyr
import Tyr.Model.KittenTTS
import Tyr.Model.KittenTTS.Checkpoint

open torch
open torch.kittentts

private structure TensorStats where
  shape : Array UInt64
  min : Float
  max : Float
  mean : Float
  std : Float
  deriving Repr, Inhabited

private def fmin (a b : Float) : Float := if a < b then a else b
private def fmax (a b : Float) : Float := if a > b then a else b

private def usage : String :=
  "Usage: lake exe KittenTTSDebug <model_dir_or_repo> <phonemes> <voice> [speed] [seed]"

private def parseFloatLit? (s : String) : Option Float :=
  match s.splitOn "." with
  | [whole] =>
      whole.toNat?.map (·.toFloat)
  | [whole, frac] =>
      match whole.toNat?, frac.toNat? with
      | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (w.toFloat + f.toFloat / denom)
      | _, _ => none
  | _ => none

private def tensorStats (t : T #[]) : IO TensorStats := do
  let vals ← data.tensorToFloatArray' t
  if vals.isEmpty then
    pure { shape := t.runtimeShape, min := 0.0, max := 0.0, mean := 0.0, std := 0.0 }
  else
    let init := vals[0]!
    let (sum, sqSum, minV, maxV) :=
      vals.foldl
        (fun (accSum, accSq, accMin, accMax) x =>
          (accSum + x, accSq + x * x, fmin accMin x, fmax accMax x))
        (0.0, 0.0, init, init)
    let n := vals.size.toFloat
    let mean := sum / n
    let var := fmax 0.0 (sqSum / n - mean * mean)
    pure {
      shape := t.runtimeShape
      min := minV
      max := maxV
      mean := mean
      std := Float.sqrt var
    }

private def printStats (name : String) (s : TensorStats) : IO Unit := do
  IO.println s!"{name}.shape={s.shape}"
  IO.println s!"{name}.min={s.min}"
  IO.println s!"{name}.max={s.max}"
  IO.println s!"{name}.mean={s.mean}"
  IO.println s!"{name}.std={s.std}"

def main (args : List String) : IO UInt32 := do
  let (source, phonemes, voice, speed, seed) ←
    match args with
    | [source, phonemes, voice] =>
      pure (source, phonemes, voice, 1.0, (0 : UInt64))
    | [source, phonemes, voice, speedStr] =>
      match parseFloatLit? speedStr with
      | some speed => pure (source, phonemes, voice, speed, (0 : UInt64))
      | none =>
        IO.eprintln s!"Invalid speed: {speedStr}"
        IO.eprintln usage
        pure ("", "", "", 1.0, (0 : UInt64))
    | [source, phonemes, voice, speedStr, seedStr] =>
      match parseFloatLit? speedStr, seedStr.toNat? with
      | some speed, some seed => pure (source, phonemes, voice, speed, seed.toUInt64)
      | none, _ =>
        IO.eprintln s!"Invalid speed: {speedStr}"
        IO.eprintln usage
        pure ("", "", "", 1.0, (0 : UInt64))
      | _, none =>
        IO.eprintln s!"Invalid seed: {seedStr}"
        IO.eprintln usage
        pure ("", "", "", 1.0, (0 : UInt64))
    | _ =>
      IO.eprintln usage
      pure ("", "", "", 1.0, (0 : UInt64))
  if source.isEmpty then
    return 1

  let seqToCF {batch seq channels : UInt64}
      (x : T #[batch, seq, channels])
      : T #[batch, channels, seq] :=
    reshape (nn.transpose x 1 2) #[batch, channels, seq]

  let log : torch.Log.Handlers := {
    onInfo := IO.println
    onWarn := fun msg => IO.eprintln s!"warning: {msg}"
    onError := fun msg => IO.eprintln s!"error: {msg}"
  }
  let bundle ← Model.loadFromPretrained source (log := log)
  let checkpointPath := s!"{bundle.modelDir}/model.safetensors"
  let checkpointTextEncoder ← KokoroCheckpoint.text_encoder.load checkpointPath
  torch.manualSeed seed
  let out ← bundle.debugPhonemes phonemes voice speed
  let refStyle ← bundle.loadVoiceStyle voice phonemes.toList.length.toUInt64
  let decoderStyle : T #[1, bundle.cfg.styleDim] := data.slice refStyle 1 0 bundle.cfg.styleDim
  let ids := bundle.inputIdsFromPhonemes phonemes
  let seq := ids.size.toUInt64
  let inputIds0 : T #[1, seq] := reshape (data.fromInt64Array (ids.map (·.toInt64))) #[1, seq]
  let device := bundle.model.bert.embeddings.wordEmbeddings.device
  let inputIds := if inputIds0.device == device then inputIds0 else inputIds0.to device
  let emb : T #[1, seq, bundle.cfg.hiddenDim] := nn.embedding inputIds bundle.model.textEncoder.embedding
  let mut convH : T #[1, bundle.cfg.hiddenDim, seq] := seqToCF emb
  let embStats ← tensorStats (nn.eraseShape emb)
  printStats "embedding" embStats
  let some checkpointConv0 := checkpointTextEncoder.cnn[0]?
    | throw <| IO.userError "checkpoint text encoder missing conv block 0"
  let some checkpointConv1 := checkpointTextEncoder.cnn[1]?
    | throw <| IO.userError "checkpoint text encoder missing conv block 1"
  let some checkpointConv2 := checkpointTextEncoder.cnn[2]?
    | throw <| IO.userError "checkpoint text encoder missing conv block 2"
  let directGamma0 ← KokoroCheckpoint.load_text_encoder_cnn_0_1_gamma checkpointPath
  let directGamma1 ← KokoroCheckpoint.load_text_encoder_cnn_1_1_gamma checkpointPath
  let directGamma2 ← KokoroCheckpoint.load_text_encoder_cnn_2_1_gamma checkpointPath
  let ckptLnWeight0Stats ← tensorStats (nn.eraseShape checkpointConv0.i1.gamma)
  let ckptLnWeight1Stats ← tensorStats (nn.eraseShape checkpointConv1.i1.gamma)
  let ckptLnWeight2Stats ← tensorStats (nn.eraseShape checkpointConv2.i1.gamma)
  let directLnWeight0Stats ← tensorStats (nn.eraseShape directGamma0)
  let directLnWeight1Stats ← tensorStats (nn.eraseShape directGamma1)
  let directLnWeight2Stats ← tensorStats (nn.eraseShape directGamma2)
  printStats "checkpoint_text_conv_0.ln_weight" ckptLnWeight0Stats
  printStats "checkpoint_text_conv_1.ln_weight" ckptLnWeight1Stats
  printStats "checkpoint_text_conv_2.ln_weight" ckptLnWeight2Stats
  printStats "direct_text_conv_0.ln_weight" directLnWeight0Stats
  printStats "direct_text_conv_1.ln_weight" directLnWeight1Stats
  printStats "direct_text_conv_2.ln_weight" directLnWeight2Stats
  let mut convIdx : Nat := 0
  for conv in bundle.model.textEncoder.convs do
    let weightStats ← tensorStats (nn.eraseShape conv.conv.weight)
    let lnWeightStats ← tensorStats (nn.eraseShape conv.lnWeight)
    let lnBiasStats ← tensorStats (nn.eraseShape conv.lnBias)
    printStats s!"text_conv_{convIdx}.weight" weightStats
    printStats s!"text_conv_{convIdx}.ln_weight" lnWeightStats
    printStats s!"text_conv_{convIdx}.ln_bias" lnBiasStats
    convH := conv.forward convH
    let convStatsI ← tensorStats (nn.eraseShape convH)
    printStats s!"text_conv_{convIdx}" convStatsI
    convIdx := convIdx + 1
  let textEnc : T #[1, bundle.cfg.hiddenDim, seq] := bundle.model.textEncoder.forward inputIds
  IO.println s!"seed={seed}"
  IO.println s!"pred_durations={out.predDurations}"
  IO.println s!"alignment_frames={out.alignmentFrames}"
  let textEncStats ← tensorStats (nn.eraseShape textEnc)
  let asrStats ← tensorStats out.asr
  let f0Stats ← tensorStats out.f0Curve
  let nStats ← tensorStats out.nCurve
  let audioStats ← tensorStats out.audio
  printStats "text_enc" textEncStats
  printStats "asr" asrStats
  printStats "f0" f0Stats
  printStats "n" nStats
  if out.alignmentFrames > 0 then
    let frames := out.alignmentFrames
    let asr : T #[1, bundle.cfg.hiddenDim, frames] := reshape out.asr #[1, bundle.cfg.hiddenDim, frames]
    let f0Curve : T #[1, 1, 2 * frames] := reshape out.f0Curve #[1, 1, 2 * frames]
    let nCurve : T #[1, 1, 2 * frames] := reshape out.nCurve #[1, 1, 2 * frames]
    let decoderDebug ← bundle.model.decoder.debugForward asr f0Curve nCurve decoderStyle
    let decoderF0Stats ← tensorStats decoderDebug.f0
    let decoderNStats ← tensorStats decoderDebug.n
    let decoderEncodeStats ← tensorStats decoderDebug.encode
    let decoderAsrResStats ← tensorStats decoderDebug.asrRes
    let decoderD0Stats ← tensorStats decoderDebug.decode0
    let decoderD1Stats ← tensorStats decoderDebug.decode1
    let decoderD2Stats ← tensorStats decoderDebug.decode2
    let decoderD3Stats ← tensorStats decoderDebug.decode3
    printStats "decoder_f0" decoderF0Stats
    printStats "decoder_n" decoderNStats
    printStats "decoder_encode" decoderEncodeStats
    printStats "decoder_asr_res" decoderAsrResStats
    printStats "decoder_decode0" decoderD0Stats
    printStats "decoder_decode1" decoderD1Stats
    printStats "decoder_decode2" decoderD2Stats
    printStats "decoder_decode3" decoderD3Stats
    let generatorHarSourceStats ← tensorStats decoderDebug.generator.harSource
    let generatorHarStats ← tensorStats decoderDebug.generator.har
    let generatorPostStats ← tensorStats decoderDebug.generator.post
    let generatorSpecLogStats ← tensorStats decoderDebug.generator.specLog
    let generatorPhaseRawStats ← tensorStats decoderDebug.generator.phaseRaw
    let generatorSpecStats ← tensorStats decoderDebug.generator.spec
    printStats "generator_har_source" generatorHarSourceStats
    printStats "generator_har" generatorHarStats
    for i in [:decoderDebug.generator.stages.size] do
      let stage := decoderDebug.generator.stages[i]!
      printStats s!"generator_stage_{i}.x_source" (← tensorStats stage.xSource)
      printStats s!"generator_stage_{i}.x_up" (← tensorStats stage.xUp)
      printStats s!"generator_stage_{i}.x_mix" (← tensorStats stage.xMix)
      printStats s!"generator_stage_{i}.x_out" (← tensorStats stage.xOut)
    printStats "generator_post" generatorPostStats
    printStats "generator_spec_log" generatorSpecLogStats
    printStats "generator_phase_raw" generatorPhaseRawStats
    printStats "generator_spec" generatorSpecStats
    printStats "generator_audio" (← tensorStats decoderDebug.generator.audio)
  printStats "audio" audioStats
  pure 0
