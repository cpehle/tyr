import Tyr.Model.Qwen3ASR
import Tyr.Tokenizer.Qwen3

open torch.qwen3asr

namespace Examples.Qwen3ASR

structure Args where
  source : String := "weights/qwen3-asr-0.6b"
  alignerSource : Option String := none
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  wavPath : String := "MLKDream.wav"
  language : Option String := none
  returnTimeStamps : Bool := false
  maxInferenceBatchSize : Int := -1
  maxNewTokens : UInt64 := 1024
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 := do
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def parseIntArg (name : String) (v : String) : IO Int := do
  match v.toInt? with
  | some n => pure n
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest =>
      parseArgsLoop rest { acc with source := v }
  | "--aligner-source" :: v :: rest =>
      parseArgsLoop rest { acc with alignerSource := some v }
  | "--model-dir" :: v :: rest =>
      parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest =>
      parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest =>
      parseArgsLoop rest { acc with cacheDir := v }
  | "--wav-path" :: v :: rest =>
      parseArgsLoop rest { acc with wavPath := v }
  | "--language" :: v :: rest =>
      parseArgsLoop rest { acc with language := some v }
  | "--return-timestamps" :: rest =>
      parseArgsLoop rest { acc with returnTimeStamps := true }
  | "--max-inference-batch-size" :: v :: rest =>
      parseArgsLoop rest { acc with maxInferenceBatchSize := (← parseIntArg "--max-inference-batch-size" v) }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--help" :: _ =>
      IO.println "Usage: lake exe Qwen3ASRTranscribe [options]"
      IO.println "  --source <path-or-repo>  Local model dir or HF repo id"
      IO.println "  --aligner-source <src>   Optional forced-aligner model dir/repo id"
      IO.println "  --model-dir <path>       Alias for --source (backward compatible)"
      IO.println "  --revision <rev>         HF revision/branch/tag (default: main)"
      IO.println "  --cache-dir <path>       Local cache for downloaded files"
      IO.println "  --wav-path <path>        WAV path to transcribe"
      IO.println "  --language <name>        Optional forced language (e.g. English)"
      IO.println "  --return-timestamps      Enable forced-alignment timestamps"
      IO.println "  --max-inference-batch-size <n>  Batch chunking control (-1 = unbounded)"
      IO.println "  --max-new-tokens <n>     Greedy decode max new tokens"
      IO.println "Example: lake exe Qwen3ASRTranscribe --source Qwen/Qwen3-ASR-1.7B --wav-path MLKDream.wav"
      throw <| IO.userError ""
  | x :: _ =>
      throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  let modelDir ← hub.resolvePretrainedDir args.source {
    revision := args.revision
    cacheDir := args.cacheDir
    includeTokenizer := true
    includePreprocessor := true
  }
  IO.println s!"Resolved model dir: {modelDir}"
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir modelDir
  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let pre ← PreprocessorConfig.loadFromPretrainedDir modelDir
  let model ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg
  let forcedAligner ←
    match args.alignerSource with
    | some src =>
      pure (some (← Qwen3ForcedAligner.loadFromPretrained src args.revision args.cacheDir))
    | none =>
      pure none
  let outs ← model.transcribeWavs
    tok
    pre
    #[args.wavPath]
    (forcedAligner := forcedAligner)
    (contexts := #[""])
    (languages := #[args.language])
    (maxInferenceBatchSize := args.maxInferenceBatchSize)
    (returnTimeStamps := args.returnTimeStamps)
    (maxNewTokens := args.maxNewTokens)
  let out := outs.getD 0 default
  IO.println s!"LANG={out.language}"
  IO.println "TEXT_BEGIN"
  IO.println out.text
  IO.println "TEXT_END"
  if args.returnTimeStamps then
    IO.println "TIMESTAMPS_BEGIN"
    match out.timeStamps with
    | some ts =>
      for it in ts.items do
        IO.println s!"{it.text}\t{it.startTime}\t{it.endTime}"
    | none =>
      IO.println ""
    IO.println "TIMESTAMPS_END"
  pure 0

end Examples.Qwen3ASR

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen3ASR.runMain argv
