/- 
  examples/Qwen3TTS/EndToEnd.lean

  End-to-end Qwen3-TTS demo in Lean:
  - load config + real checkpoint weights
  - tokenize text prompt
  - generate codec tokens with Lean talker
  - decode codec tokens to waveform with Lean speech-tokenizer decoder
-/
import Tyr.Torch
import Tyr.Model.Qwen3TTS
import Tyr.Tokenizer

namespace Examples.Qwen3TTS

open torch
open torch.qwen3tts
open tokenizer.qwen3

structure Args where
  modelDir : String := "weights/qwen3-tts"
  text : String := "Hello from Lean Qwen3-TTS."
  styleText : Option String := none
  maxTextLen : Nat := 512
  maxFrames : Nat := 256
  language : String := "Auto"
  thinkMode : String := "auto"
  seed : Option UInt64 := none
  temperature : Float := 0.9
  topK : Nat := 50
  topP : Float := 1.0
  repetitionPenalty : Float := 1.05
  suppressTail : Nat := 1024
  subtalkerTemperature : Float := 0.9
  subtalkerTopK : Nat := 50
  subtalkerTopP : Float := 1.0
  subtalkerTemperaturesByGroup : Option (Array Float) := none
  subtalkerTopKsByGroup : Option (Array Nat) := none
  subtalkerTopPsByGroup : Option (Array Float) := none
  encodeAudioPath : Option String := none
  encodeOutCodesPath : String := "output/qwen3tts_encoded_codes.txt"
  encodeOnly : Bool := false
  codesPath : String := "output/qwen3tts_codes.txt"
  wavPath : String := "output/qwen3tts.wav"
  skipDecode : Bool := false
  pythonExe : String := "uv"
  decodeScript : String := "scripts/qwen3tts_decode_codes.py"
  encodeScript : String := "scripts/qwen3tts_encode_audio.py"
  speakerMelScript : String := "scripts/qwen3tts_prepare_speaker_mel.py"
  deviceMap : Option String := none
  speakerEmbeddingPath : Option String := none
  refAudioPath : Option String := none
  speakerMelPath : String := ""
  speakerMelFramesPath : String := ""
  qwenRepo : String := "../Qwen3-TTS"
  speechTokenizerDir : Option String := none
  decodeViaPython : Bool := false
  streamingDecode : Bool := false
  trueStreaming : Bool := false
  streamChunkFrames : Nat := 4
  decodeChunkSize : Nat := 300
  decodeLeftContext : Nat := 25
  tokenIdsPath : String := "output/qwen3tts_text_ids.txt"
  help : Bool := false
  deriving Repr, Inhabited

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

private def parseFloatCsv? (s : String) : Option (Array Float) := Id.run do
  let mut out : Array Float := #[]
  for part in s.splitOn "," do
    let tok := part.trimAscii.toString
    if tok.isEmpty then
      return none
    match parseFloatLit? tok with
    | some v => out := out.push v
    | none => return none
  if out.isEmpty then none else some out

private def parseNatCsv? (s : String) : Option (Array Nat) := Id.run do
  let mut out : Array Nat := #[]
  for part in s.splitOn "," do
    let tok := part.trimAscii.toString
    if tok.isEmpty then
      return none
    match tok.toNat? with
    | some v => out := out.push v
    | none => return none
  if out.isEmpty then none else some out

private inductive ThinkMode where
  | auto
  | think
  | noThink
  deriving Repr, Inhabited, BEq

private partial def parseArgsLoop : List String → Args → Args
  | [], acc => acc
  | "--model-dir" :: v :: rest, acc => parseArgsLoop rest { acc with modelDir := v }
  | "--text" :: v :: rest, acc => parseArgsLoop rest { acc with text := v }
  | "--style-text" :: v :: rest, acc => parseArgsLoop rest { acc with styleText := some v }
  | "--max-text-len" :: v :: rest, acc =>
      parseArgsLoop rest { acc with maxTextLen := v.toNat?.getD acc.maxTextLen }
  | "--max-frames" :: v :: rest, acc =>
      parseArgsLoop rest { acc with maxFrames := v.toNat?.getD acc.maxFrames }
  | "--language" :: v :: rest, acc => parseArgsLoop rest { acc with language := v }
  | "--think-mode" :: v :: rest, acc => parseArgsLoop rest { acc with thinkMode := v }
  | "--seed" :: v :: rest, acc =>
      parseArgsLoop rest { acc with seed := v.toNat?.map UInt64.ofNat }
  | "--temperature" :: v :: rest, acc =>
      parseArgsLoop rest { acc with temperature := (parseFloatLit? v).getD acc.temperature }
  | "--top-k" :: v :: rest, acc =>
      parseArgsLoop rest { acc with topK := v.toNat?.getD acc.topK }
  | "--top-p" :: v :: rest, acc =>
      parseArgsLoop rest { acc with topP := (parseFloatLit? v).getD acc.topP }
  | "--repetition-penalty" :: v :: rest, acc =>
      parseArgsLoop rest { acc with repetitionPenalty := (parseFloatLit? v).getD acc.repetitionPenalty }
  | "--suppress-tail" :: v :: rest, acc =>
      parseArgsLoop rest { acc with suppressTail := v.toNat?.getD acc.suppressTail }
  | "--subtalker-temperature" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTemperature := (parseFloatLit? v).getD acc.subtalkerTemperature }
  | "--subtalker-top-k" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTopK := v.toNat?.getD acc.subtalkerTopK }
  | "--subtalker-top-p" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTopP := (parseFloatLit? v).getD acc.subtalkerTopP }
  | "--subtalker-temp-by-group" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTemperaturesByGroup := parseFloatCsv? v }
  | "--subtalker-top-k-by-group" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTopKsByGroup := parseNatCsv? v }
  | "--subtalker-top-p-by-group" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTopPsByGroup := parseFloatCsv? v }
  | "--encode-audio-path" :: v :: rest, acc => parseArgsLoop rest { acc with encodeAudioPath := some v }
  | "--encode-out-codes-path" :: v :: rest, acc => parseArgsLoop rest { acc with encodeOutCodesPath := v }
  | "--encode-only" :: rest, acc => parseArgsLoop rest { acc with encodeOnly := true }
  | "--codes-path" :: v :: rest, acc => parseArgsLoop rest { acc with codesPath := v }
  | "--wav-path" :: v :: rest, acc => parseArgsLoop rest { acc with wavPath := v }
  | "--python" :: v :: rest, acc => parseArgsLoop rest { acc with pythonExe := v }
  | "--decode-script" :: v :: rest, acc => parseArgsLoop rest { acc with decodeScript := v }
  | "--encode-script" :: v :: rest, acc => parseArgsLoop rest { acc with encodeScript := v }
  | "--speaker-mel-script" :: v :: rest, acc => parseArgsLoop rest { acc with speakerMelScript := v }
  | "--device-map" :: v :: rest, acc => parseArgsLoop rest { acc with deviceMap := some v }
  | "--speaker-embedding-path" :: v :: rest, acc =>
      parseArgsLoop rest { acc with speakerEmbeddingPath := some v }
  | "--ref-audio-path" :: v :: rest, acc =>
      parseArgsLoop rest { acc with refAudioPath := some v }
  | "--speaker-mel-path" :: v :: rest, acc => parseArgsLoop rest { acc with speakerMelPath := v }
  | "--speaker-mel-frames-path" :: v :: rest, acc =>
      parseArgsLoop rest { acc with speakerMelFramesPath := v }
  | "--qwen-repo" :: v :: rest, acc => parseArgsLoop rest { acc with qwenRepo := v }
  | "--speech-tokenizer-dir" :: v :: rest, acc =>
      parseArgsLoop rest { acc with speechTokenizerDir := some v }
  | "--decode-via-python" :: rest, acc =>
      parseArgsLoop rest { acc with decodeViaPython := true }
  | "--streaming-decode" :: rest, acc =>
      parseArgsLoop rest { acc with streamingDecode := true }
  | "--true-streaming" :: rest, acc =>
      parseArgsLoop rest { acc with trueStreaming := true }
  | "--stream-chunk-frames" :: v :: rest, acc =>
      parseArgsLoop rest { acc with streamChunkFrames := v.toNat?.getD acc.streamChunkFrames }
  | "--decode-chunk-size" :: v :: rest, acc =>
      parseArgsLoop rest { acc with decodeChunkSize := v.toNat?.getD acc.decodeChunkSize }
  | "--decode-left-context" :: v :: rest, acc =>
      parseArgsLoop rest { acc with decodeLeftContext := v.toNat?.getD acc.decodeLeftContext }
  | "--token-ids-path" :: v :: rest, acc => parseArgsLoop rest { acc with tokenIdsPath := v }
  | "--skip-decode" :: rest, acc => parseArgsLoop rest { acc with skipDecode := true }
  | "--help" :: rest, acc => parseArgsLoop rest { acc with help := true }
  | _ :: rest, acc => parseArgsLoop rest acc

private def parseArgs (raw : List String) : Args :=
  parseArgsLoop raw {}

private def printUsage : IO Unit := do
  IO.println "Usage: lake exe Qwen3TTSEndToEnd [options]"
  IO.println ""
  IO.println "Options:"
  IO.println "  --model-dir <path>            Qwen3-TTS model directory (contains config.json + sharded weights)"
  IO.println "  --text <prompt>               Text prompt"
  IO.println "  --style-text <prompt>         Optional style prefix text prepended to prompt"
  IO.println "  --max-text-len <n>            Max tokenized prompt length (default: 512)"
  IO.println "  --max-frames <n>              Max generated codec frames (default: 256)"
  IO.println "  --language <name|Auto>        Language conditioning tag (default: Auto)"
  IO.println "  --think-mode <auto|think|no-think>  Reasoning-mode codec prefill control (default: auto)"
  IO.println "  --seed <u64>                  Deterministic RNG seed for sampling"
  IO.println "  --temperature <f>             Sampling temperature (default: 0.9)"
  IO.println "  --top-k <n>                   Top-k sampling (default: 50)"
  IO.println "  --top-p <f>                   Top-p sampling (default: 1.0)"
  IO.println "  --repetition-penalty <f>      Repetition penalty for first-codebook sampling (default: 1.05)"
  IO.println "  --suppress-tail <n>           Suppress top tail vocab IDs except EOS (default: 1024)"
  IO.println "  --subtalker-temperature <f>   Subtalker sampling temperature (default: 0.9)"
  IO.println "  --subtalker-top-k <n>         Subtalker top-k sampling (default: 50)"
  IO.println "  --subtalker-top-p <f>         Subtalker top-p sampling (default: 1.0)"
  IO.println "  --subtalker-temp-by-group <csv>  Residual-group temperatures, e.g. 0.9,0.85,0.8"
  IO.println "  --subtalker-top-k-by-group <csv> Residual-group top-k values, e.g. 50,40,30"
  IO.println "  --subtalker-top-p-by-group <csv> Residual-group top-p values, e.g. 1.0,0.95,0.9"
  IO.println "  --encode-audio-path <path>    Encode audio to speech-tokenizer codec IDs"
  IO.println "  --encode-out-codes-path <p>   Output codec-ID file for --encode-audio-path"
  IO.println "  --encode-only                 Run audio->codes bridge and exit"
  IO.println "  --codes-path <path>           Output codec-token text file"
  IO.println "  --wav-path <path>             Output wav path"
  IO.println "  --skip-decode                 Skip Python decode step (codes only)"
  IO.println "  --python <exe>                Python launcher (default: uv; uses `uv run python`)"
  IO.println "  --decode-script <path>        Codec decode bridge script path"
  IO.println "  --encode-script <path>        Codec encode bridge script path"
  IO.println "  --speaker-mel-script <path>   Speaker mel extraction script path"
  IO.println "  --device-map <v>              Optional HF device_map for speech tokenizer bridge"
  IO.println "  --speaker-embedding-path <p>  Optional speaker embedding tensor path ([enc_dim])"
  IO.println "  --ref-audio-path <path>       Optional reference audio to derive speaker embedding"
  IO.println "  --speaker-mel-path <path>     Optional debug dump: intermediate speaker mel safetensors"
  IO.println "  --speaker-mel-frames-path <p> Optional debug dump: intermediate mel frame-count file"
  IO.println "  --token-ids-path <path>       Output token IDs file from Lean tokenizer"
  IO.println "  --qwen-repo <path>            Local Qwen3-TTS repo path (fallback import)"
  IO.println "  --speech-tokenizer-dir <path> Speech tokenizer dir (default: <model-dir>/speech_tokenizer)"
  IO.println "  --decode-via-python           Use legacy Python decode bridge instead of Lean decoder"
  IO.println "  --streaming-decode            Use chunked Lean decoder for streaming/long-form decode"
  IO.println "  --true-streaming              Online frame-by-frame generation + incremental Lean decode"
  IO.println "  --stream-chunk-frames <n>     Frame chunk size emitted by true streaming decoder (default: 4)"
  IO.println "  --decode-chunk-size <n>       Codec-frame chunk size for --streaming-decode (default: 300)"
  IO.println "  --decode-left-context <n>     Left context codec frames for chunked decode (default: 25)"
  IO.println "  --help                        Show this help"

private def expandHome (path : String) : IO String := do
  if path == "~" then
    return (← IO.getEnv "HOME").getD path
  else if path.startsWith "~/" then
    return s!"{(← IO.getEnv "HOME").getD ""}/{path.drop 2}"
  else
    return path

private def ensureParentDir (path : String) : IO Unit := do
  match System.FilePath.parent ⟨path⟩ with
  | some parent =>
      if parent.toString != "" && parent.toString != "." then
        IO.FS.createDirAll parent
  | none => pure ()

private def formatCodesRows (flat : Array UInt64) (numCodeGroups : UInt64) : String := Id.run do
  let g := numCodeGroups.toNat
  if g == 0 then
    return ""
  let rows := flat.size / g
  let mut out := ""
  for r in [:rows] do
    for c in [:g] do
      if c > 0 then
        out := out ++ " "
      out := out ++ toString (flat[r * g + c]!)
    out := out ++ "\n"
  out

private def formatCodesMatrix (rows : Array (Array UInt64)) : String := Id.run do
  let mut out := ""
  for row in rows do
    out := out ++ String.intercalate " " (row.toList.map toString) ++ "\n"
  out

private def lowerAsciiString (s : String) : String :=
  String.ofList <| s.toList.map (fun c =>
    if c >= 'A' && c <= 'Z' then Char.ofNat (c.toNat + 32) else c)

private def parseThinkMode (s : String) : ThinkMode :=
  match lowerAsciiString s with
  | "think" => .think
  | "no-think" => .noThink
  | "nothink" => .noThink
  | "no_think" => .noThink
  | _ => .auto

private def lookupLanguageId? (cfg : Qwen3TTSConfig) (language : String) : Option UInt64 := Id.run do
  let q := lowerAsciiString language
  for (k, v) in cfg.talkerConfig.codecLanguageId do
    if lowerAsciiString k == q then
      return some v
  none

private def idsTensor1 (ids : Array UInt64) : T #[1, ids.size.toUInt64] :=
  reshape (data.fromInt64Array (ids.map (Int64.ofNat ∘ UInt64.toNat))) #[1, ids.size.toUInt64]

private def saveTokenIds (path : String) (ids : Array UInt64) : IO Unit := do
  ensureParentDir path
  let text := String.intercalate " " (ids.toList.map toString)
  IO.FS.writeFile path (text ++ "\n")

private def loadSpeakerEmbedding? (cfg : Qwen3TTSConfig) (path? : Option String)
    : IO (Option (T #[1, 1, cfg.talkerConfig.hiddenSize])) := do
  match path? with
  | none => pure none
  | some p =>
      if cfg.speakerEncoderConfig.encDim != cfg.talkerConfig.hiddenSize then
        throw <| IO.userError
          s!"speaker embedding dim mismatch: speaker enc_dim={cfg.speakerEncoderConfig.encDim}, talker hidden={cfg.talkerConfig.hiddenSize}"
      let path ← expandHome p
      let emb : T #[cfg.speakerEncoderConfig.encDim] ← data.loadTensor #[cfg.speakerEncoderConfig.encDim] path
      pure (some (reshape emb #[1, 1, cfg.talkerConfig.hiddenSize]))

private def buildTalkerInputsEquivalent
    (cfg : Qwen3TTSConfig)
    (model : Qwen3TTSForConditionalGeneration cfg)
    (inputIds : Array UInt64)
    (languageId : Option UInt64)
    (thinkMode : ThinkMode := .auto)
    (speakerEmbed : Option (T #[1, 1, cfg.talkerConfig.hiddenSize]) := none)
    : IO ((Sigma fun seq => T #[1, seq, cfg.talkerConfig.hiddenSize]) × T #[1, 1, cfg.talkerConfig.hiddenSize]) := do
  if inputIds.size < 8 then
    throw <| IO.userError s!"Tokenized assistant input is too short ({inputIds.size}); expected at least 8 tokens."

  let roleIds := inputIds.extract 0 3
  if roleIds.size != 3 then
    throw <| IO.userError s!"Unexpected role-token size: {roleIds.size} (expected 3)."
  let textMainIds := inputIds.extract 3 (inputIds.size - 5)
  let textMainLen : UInt64 := textMainIds.size.toUInt64
  let hidden : UInt64 := cfg.talkerConfig.hiddenSize

  let roleTensor : T #[1, 3] := idsTensor1 roleIds
  let roleEmbed : T #[1, 3, hidden] := TalkerModel.embedText model.talker.model roleTensor

  let ttsSpecialIds : T #[1, 3] := reshape (data.fromInt64Array #[
    Int64.ofNat cfg.ttsBosTokenId.toNat,
    Int64.ofNat cfg.ttsEosTokenId.toNat,
    Int64.ofNat cfg.ttsPadTokenId.toNat
  ]) #[1, 3]
  let ttsSpecialEmb : T #[1, 3, hidden] := TalkerModel.embedText model.talker.model ttsSpecialIds
  let ttsBosEmbed : T #[1, 1, hidden] := data.slice ttsSpecialEmb 1 0 1
  let ttsEosEmbed : T #[1, 1, hidden] := data.slice ttsSpecialEmb 1 1 1
  let ttsPadEmbed : T #[1, 1, hidden] := data.slice ttsSpecialEmb 1 2 1

  let codecPrefillIds : Array UInt64 :=
    match thinkMode, languageId with
    | .noThink, _ =>
        #[
          cfg.talkerConfig.codecNoThinkId,
          cfg.talkerConfig.codecThinkBosId,
          cfg.talkerConfig.codecThinkEosId
        ]
    | .think, some langId =>
        #[
          cfg.talkerConfig.codecThinkId,
          cfg.talkerConfig.codecThinkBosId,
          langId,
          cfg.talkerConfig.codecThinkEosId
        ]
    | .think, none =>
        #[
          cfg.talkerConfig.codecThinkId,
          cfg.talkerConfig.codecThinkBosId,
          cfg.talkerConfig.codecThinkEosId
        ]
    | .auto, some langId =>
        #[
          cfg.talkerConfig.codecThinkId,
          cfg.talkerConfig.codecThinkBosId,
          langId,
          cfg.talkerConfig.codecThinkEosId
        ]
    | .auto, none =>
        #[
          cfg.talkerConfig.codecNoThinkId,
          cfg.talkerConfig.codecThinkBosId,
          cfg.talkerConfig.codecThinkEosId
        ]
  let codecPrefillLen : UInt64 := codecPrefillIds.size.toUInt64
  let codecPrefillTensor : T #[1, codecPrefillLen] := idsTensor1 codecPrefillIds
  let codecEmb0 : T #[1, codecPrefillLen, hidden] := TalkerModel.embedCodec model.talker.model codecPrefillTensor

  let codecPadBosIds : T #[1, 2] := reshape (data.fromInt64Array #[
    Int64.ofNat cfg.talkerConfig.codecPadId.toNat,
    Int64.ofNat cfg.talkerConfig.codecBosId.toNat
  ]) #[1, 2]
  let codecEmb1 : T #[1, 2, hidden] := TalkerModel.embedCodec model.talker.model codecPadBosIds

  let (codecInputLen, codecInput) ←
    match speakerEmbed with
    | some spk =>
        let codecWithSpeaker : T #[1, codecPrefillLen + 1, hidden] := nn.cat codecEmb0 spk 1
        let codecInput : T #[1, (codecPrefillLen + 1) + 2, hidden] := nn.cat codecWithSpeaker codecEmb1 1
        pure (((codecPrefillLen + 1) + 2), codecInput)
    | none =>
        let codecInput : T #[1, codecPrefillLen + 2, hidden] := nn.cat codecEmb0 codecEmb1 1
        pure ((codecPrefillLen + 2), codecInput)

  let codecPrefix : T #[1, codecInputLen - 1, hidden] := data.slice codecInput 1 0 (codecInputLen - 1)
  let padScaffold : T #[1, codecInputLen - 2, hidden] := nn.expand ttsPadEmbed #[1, codecInputLen - 2, hidden]
  let scaffold0 : T #[1, codecInputLen - 1, hidden] :=
    reshape (nn.cat padScaffold ttsBosEmbed 1) #[1, codecInputLen - 1, hidden]
  let scaffold : T #[1, codecInputLen - 1, hidden] := scaffold0 + codecPrefix
  let partA : T #[1, 3 + (codecInputLen - 1), hidden] := nn.cat roleEmbed scaffold 1

  let textMainTensor : T #[1, textMainLen] := idsTensor1 textMainIds
  let textMainEmbed : T #[1, textMainLen, hidden] := TalkerModel.embedText model.talker.model textMainTensor
  let textPlusEos : T #[1, textMainLen + 1, hidden] := nn.cat textMainEmbed ttsEosEmbed 1
  let codecPadIds : T #[1, textMainLen + 1] :=
    torch.full_int #[1, textMainLen + 1] (Int64.ofNat cfg.talkerConfig.codecPadId.toNat)
  let codecPadEmbed : T #[1, textMainLen + 1, hidden] := TalkerModel.embedCodec model.talker.model codecPadIds
  let partB : T #[1, textMainLen + 1, hidden] := textPlusEos + codecPadEmbed

  let codecBosOnlyIds : T #[1, 1] := torch.full_int #[1, 1] (Int64.ofNat cfg.talkerConfig.codecBosId.toNat)
  let codecBosEmbed : T #[1, 1, hidden] := TalkerModel.embedCodec model.talker.model codecBosOnlyIds
  let partC : T #[1, 1, hidden] := ttsPadEmbed + codecBosEmbed

  let partABLen : UInt64 := (3 + (codecInputLen - 1)) + (textMainLen + 1)
  let partAB : T #[1, partABLen, hidden] := nn.cat partA partB 1
  let outLen : UInt64 := partABLen + 1
  let talkerInputs : T #[1, outLen, hidden] := nn.cat partAB partC 1
  pure ((⟨outLen, talkerInputs⟩), ttsPadEmbed)

private def buildBridgeConfig (args : Args) : SpeechTokenizerBridgeConfig := {
  pythonExe := args.pythonExe
  qwenRepo := args.qwenRepo
  speechTokenizerDir := args.speechTokenizerDir
  decodeScript := args.decodeScript
  encodeScript := args.encodeScript
  speakerMelScript := args.speakerMelScript
  deviceMap := args.deviceMap
}

private def resolveRuntimeDevice : IO Device := do
  let requested := (← IO.getEnv "TYR_DEVICE").map String.toLower
  match requested with
  | some "cpu" => pure Device.CPU
  | some "cuda" => pure (Device.CUDA 0)
  | some "mps" => pure Device.MPS
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none => getBestDevice

def runEndToEnd (args : Args) : IO Unit := do
  let modelDir ← expandHome args.modelDir
  let codesPath ← expandHome args.codesPath
  let encodeOutCodesPath ← expandHome args.encodeOutCodesPath
  let wavPath ← expandHome args.wavPath
  let tokenIdsPath ← expandHome args.tokenIdsPath
  let speakerMelPath ← expandHome args.speakerMelPath
  let speakerMelFramesPath ← expandHome args.speakerMelFramesPath
  let targetDevice ← resolveRuntimeDevice
  let bridgeCfgBase := buildBridgeConfig args
  let bridgeCfg : SpeechTokenizerBridgeConfig :=
    match bridgeCfgBase.deviceMap with
    | some _ => bridgeCfgBase
    | none =>
        match targetDevice with
        | Device.MPS => { bridgeCfgBase with deviceMap := some "mps" }
        | Device.CUDA _ => { bridgeCfgBase with deviceMap := some "cuda" }
        | _ => bridgeCfgBase

  IO.println "=== Qwen3-TTS End-to-End (Lean) ==="
  IO.println s!"Model dir: {modelDir}"
  IO.println s!"Prompt: {args.text}"
  IO.println s!"Language: {args.language}"
  IO.println s!"Target device: {repr targetDevice}"
  let thinkMode := parseThinkMode args.thinkMode
  let subtalkerTopKsByGroup : Option (Array UInt64) :=
    args.subtalkerTopKsByGroup.map (fun xs => xs.map Nat.toUInt64)
  match args.seed with
  | some seed =>
      torch.manualSeed seed
      IO.println s!"Sampling seed: {seed}"
  | none =>
      pure ()

  -- Load runtime config from HF config.json so tensor shapes match real checkpoints.
  let cfg ← Qwen3TTSConfig.loadFromPretrainedDir modelDir
  IO.println s!"Model type: {cfg.ttsModelType}, talker hidden={cfg.talkerConfig.hiddenSize}, code groups={cfg.talkerConfig.numCodeGroups}"

  match args.encodeAudioPath with
  | some audioPath =>
      let audioPath ← expandHome audioPath
      encodeAudioToCodes bridgeCfg modelDir audioPath encodeOutCodesPath targetDevice
      IO.println s!"Saved encoded audio codec IDs to {encodeOutCodesPath}"
      if args.encodeOnly then
        IO.println "Skipping TTS generation (--encode-only enabled)."
        return
  | none =>
      if args.encodeOnly then
        throw <| IO.userError "--encode-only requires --encode-audio-path"

  let model ← Qwen3TTSForConditionalGeneration.loadSharded modelDir cfg targetDevice

  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let baseText :=
    match args.styleText with
    | some style =>
        let style := style.trimAscii.toString
        if style.isEmpty then
          args.text
        else
          s!"{style}\n{args.text}"
    | none =>
        args.text
  let assistantText := tokenizer.qwen3.ttsAssistantText baseText
  let rawTokenIds := (tokenizer.qwen3.encodeText tok assistantText).map (fun t => t.toUInt64)
  let tokenIds :=
    if rawTokenIds.size > args.maxTextLen then
      rawTokenIds.extract 0 args.maxTextLen
    else
      rawTokenIds
  saveTokenIds tokenIdsPath tokenIds
  let textTokenCount := tokenIds.size.toUInt64
  if textTokenCount == 0 then
    throw <| IO.userError "Tokenized prompt is empty."
  IO.println s!"Tokenized prompt length: {textTokenCount}"
  IO.println s!"Saved token IDs to {tokenIdsPath}"

  let languageId ←
    if lowerAsciiString args.language == "auto" then
      pure none
    else
      match lookupLanguageId? cfg args.language with
      | some lid => pure (some lid)
      | none =>
          let supported := String.intercalate ", " (cfg.talkerConfig.codecLanguageId.toList.map (·.fst))
          throw <| IO.userError s!"Unsupported language '{args.language}'. Supported: {supported}"

  let speakerEmbed ←
    match args.speakerEmbeddingPath, args.refAudioPath with
    | some p, _ =>
        loadSpeakerEmbedding? cfg (some p)
    | none, some refAudioPath =>
        let refAudioPath ← expandHome refAudioPath
        let emb ← extractSpeakerEmbeddingFromAudio
          cfg model bridgeCfg refAudioPath speakerMelPath speakerMelFramesPath
        IO.println s!"Extracted speaker embedding from reference audio: {refAudioPath}"
        pure (some emb)
    | none, none =>
        pure none
  let ((⟨talkerSeq, talkerInputs⟩), ttsPadEmbed) ←
    buildTalkerInputsEquivalent cfg model tokenIds languageId thinkMode speakerEmbed
  IO.println s!"Built conditioned talker inputs, seq={talkerSeq}"

  let maxFrames : UInt64 := args.maxFrames.toUInt64
  if args.trueStreaming then
    if args.decodeViaPython then
      throw <| IO.userError "--true-streaming currently requires Lean decode (remove --decode-via-python)."
    if cfg.talkerConfig.numCodeGroups != 16 then
      throw <| IO.userError
        s!"true streaming currently supports 16 code groups, got {cfg.talkerConfig.numCodeGroups}."

    ensureParentDir codesPath
    let speechDecoder? ←
      if args.skipDecode then
        pure none
      else
        let speechTokenizerDir ←
          match args.speechTokenizerDir with
          | some d => expandHome d
          | none => pure s!"{modelDir}/speech_tokenizer"
        IO.println s!"Loading Lean speech-tokenizer decoder from {speechTokenizerDir}..."
        let dec ← SpeechTokenizer12HzDecoder.loadFromDir speechTokenizerDir targetDevice
        ensureParentDir wavPath
        data.wavBegin wavPath dec.outputSampleRate
        pure (some dec)

    let codeRowsRef ← IO.mkRef (#[] : Array (Array UInt64))
    let decodeStateRef ←
      match speechDecoder? with
      | some _ =>
          let st : SpeechTokenizer12HzDecoder.DecodeStreamState 1 :=
            SpeechTokenizer12HzDecoder.initDecodeStreamState
              args.streamChunkFrames.toUInt64 args.decodeLeftContext.toUInt64 targetDevice
          IO.mkRef (some st : Option (SpeechTokenizer12HzDecoder.DecodeStreamState 1))
      | none =>
          IO.mkRef (none : Option (SpeechTokenizer12HzDecoder.DecodeStreamState 1))

    let onFrame : UInt64 → T #[1, cfg.talkerConfig.numCodeGroups] → IO Unit := fun _ frame => do
      let frameRow : T #[cfg.talkerConfig.numCodeGroups] := reshape frame #[cfg.talkerConfig.numCodeGroups]
      let rowVals ← data.tensorToUInt64Array frameRow
      let firstTok := rowVals.getD 0 cfg.talkerConfig.codecEosTokenId
      if firstTok != cfg.talkerConfig.codecEosTokenId then
        codeRowsRef.modify (fun rows => rows.push rowVals)
        match speechDecoder? with
        | some dec =>
            match (← decodeStateRef.get) with
            | some st =>
                let frame3 : T #[1, 16, 1] := reshape frame #[1, 16, 1]
                let (st', chunks) := dec.pushDecodeStream st frame3
                decodeStateRef.set (some st')
                for chunk in chunks do
                  data.wavAppend chunk wavPath
            | none => pure ()
        | none => pure ()

    let _lengths ← TalkerForConditionalGeneration.streamCodes
      cfg.talkerConfig model.talker talkerInputs onFrame maxFrames
      2
      args.temperature args.topK.toUInt64 args.topP
      args.subtalkerTemperature args.subtalkerTopK.toUInt64 args.subtalkerTopP
      args.repetitionPenalty args.suppressTail.toUInt64
      (some ⟨1, ttsPadEmbed⟩) (some ttsPadEmbed)
      args.subtalkerTemperaturesByGroup subtalkerTopKsByGroup args.subtalkerTopPsByGroup

    match speechDecoder? with
    | some dec =>
        match (← decodeStateRef.get) with
        | some st =>
            let (_stFinal, chunks) := dec.flushDecodeStream st
            for chunk in chunks do
              data.wavAppend chunk wavPath
            data.wavFinalize wavPath
            IO.println s!"Saved waveform to {wavPath} (Lean true streaming decode)"
        | none => pure ()
    | none =>
        IO.println "Skipping waveform decode (--skip-decode enabled)."

    let rows ← codeRowsRef.get
    let codeLen := rows.size.toUInt64
    let codesText := formatCodesMatrix rows
    IO.FS.writeFile codesPath codesText
    IO.println s!"Generated {codeLen} codec frames."
    IO.println s!"Saved codec codes to {codesPath}"
  else
    let out ← TalkerForConditionalGeneration.generateCodesWithLengths
      cfg.talkerConfig model.talker talkerInputs maxFrames
      2
      args.temperature args.topK.toUInt64 args.topP
      args.subtalkerTemperature args.subtalkerTopK.toUInt64 args.subtalkerTopP
      args.repetitionPenalty args.suppressTail.toUInt64
      (some ⟨1, ttsPadEmbed⟩) (some ttsPadEmbed)
      args.subtalkerTemperaturesByGroup subtalkerTopKsByGroup args.subtalkerTopPsByGroup
    let codeLen := out.lengths.getD 0 maxFrames

    let codes3 : T #[1, maxFrames, cfg.talkerConfig.numCodeGroups] := data.slice out.codes 0 0 1
    let codes2 : T #[maxFrames, cfg.talkerConfig.numCodeGroups] :=
      reshape codes3 #[maxFrames, cfg.talkerConfig.numCodeGroups]
    let trimmed : T #[codeLen, cfg.talkerConfig.numCodeGroups] := data.slice codes2 0 0 codeLen
    let flat : T #[codeLen * cfg.talkerConfig.numCodeGroups] :=
      reshape trimmed #[codeLen * cfg.talkerConfig.numCodeGroups]
    let codeVals ← data.tensorToUInt64Array flat

    ensureParentDir codesPath
    let codesText := formatCodesRows codeVals cfg.talkerConfig.numCodeGroups
    IO.FS.writeFile codesPath codesText
    IO.println s!"Generated {codeLen} codec frames."
    IO.println s!"Saved codec codes to {codesPath}"

    if args.skipDecode then
      IO.println "Skipping waveform decode (--skip-decode enabled)."
    else
      ensureParentDir wavPath
      if args.decodeViaPython then
        decodeCodesToWav bridgeCfg modelDir cfg.talkerConfig codesPath wavPath
        IO.println s!"Saved waveform to {wavPath} (Python decode bridge)"
      else
        if cfg.talkerConfig.numCodeGroups != 16 then
          throw <| IO.userError
            s!"Lean speech-tokenizer decoder currently supports 16 code groups, got {cfg.talkerConfig.numCodeGroups}. Use --decode-via-python."
        let speechTokenizerDir ←
          match args.speechTokenizerDir with
          | some d => expandHome d
          | none => pure s!"{modelDir}/speech_tokenizer"
        IO.println s!"Loading Lean speech-tokenizer decoder from {speechTokenizerDir}..."
        let speechDecoder ← SpeechTokenizer12HzDecoder.loadFromDir speechTokenizerDir targetDevice
        let trimmed16 : T #[codeLen, 16] := reshape trimmed #[codeLen, 16]
        if args.streamingDecode then
          speechDecoder.decodeFrameMajorChunkedToWav
            trimmed16 wavPath args.decodeChunkSize.toUInt64 args.decodeLeftContext.toUInt64
          IO.println s!"Saved waveform to {wavPath} (Lean decoder, chunked streaming)"
        else
          speechDecoder.decodeFrameMajorToWav trimmed16 wavPath
          IO.println s!"Saved waveform to {wavPath} (Lean decoder)"

def _root_.main (rawArgs : List String) : IO UInt32 := do
  let args := parseArgs rawArgs
  if args.help then
    printUsage
    return 0
  runEndToEnd args
  return 0

end Examples.Qwen3TTS
