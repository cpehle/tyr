import Tyr.Model.Whisper

open torch.whisper

namespace Examples.Whisper

structure Args where
  modelDir : String := "weights/whisper-tiny"
  wavPath : String := "MLKDream.wav"
  language : String := "en"
  maxNewTokens : UInt64 := 128
  noTimestamps : Bool := true
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 := do
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--model-dir" :: v :: rest =>
      parseArgsLoop rest { acc with modelDir := v }
  | "--source" :: v :: rest =>
      parseArgsLoop rest { acc with modelDir := v }
  | "--wav-path" :: v :: rest =>
      parseArgsLoop rest { acc with wavPath := v }
  | "--language" :: v :: rest =>
      parseArgsLoop rest { acc with language := v }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--no-timestamps" :: rest =>
      parseArgsLoop rest { acc with noTimestamps := true }
  | "--help" :: _ =>
      IO.println "Usage: lake exe WhisperTranscribe [options]"
      IO.println "  --model-dir <path>       Whisper model directory (HF layout with config/tokenizer/safetensors)"
      IO.println "  --source <path>          Alias for --model-dir"
      IO.println "  --wav-path <path>        WAV file path"
      IO.println "  --language <code>        Language code/name (default: en)"
      IO.println "  --max-new-tokens <n>     Max generated decoder tokens (default: 128)"
      IO.println "  --no-timestamps          Use <|notimestamps|> decoding prompt token"
      throw <| IO.userError ""
  | x :: _ =>
      throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  let bundle ← loadFromPretrainedDir args.modelDir
  let out ←
    transcribeWav
      bundle.model
      bundle.tok
      bundle.preprocessor
      args.wavPath
      args.language
      args.maxNewTokens
      args.noTimestamps
  IO.println s!"LANG={out.language}"
  IO.println "TEXT_BEGIN"
  IO.println out.text
  IO.println "TEXT_END"
  pure 0

end Examples.Whisper

def main (argv : List String) : IO UInt32 :=
  Examples.Whisper.runMain argv
