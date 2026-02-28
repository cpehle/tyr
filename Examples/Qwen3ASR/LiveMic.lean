import Tyr.Model.Qwen3ASR
import Tyr.Tokenizer.Qwen3
import Tyr.Audio.AppleInput

namespace Examples.Qwen3ASR

open torch.qwen3asr

structure Args where
  source : String := "weights/qwen3-asr-0.6b"
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  language : Option String := none
  context : String := ""
  maxNewTokens : UInt64 := 128
  chunkSec : Float := 2.0
  hopSec : Float := 0.5
  runSec : Float := 30.0
  simpleOutput : Bool := false
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 :=
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def parseFloatArg (name : String) (v : String) : IO Float :=
  if v.contains '.' then
    let parts := v.splitOn "."
    let intPart := parts.getD 0 "" |>.toNat?.getD 0
    let fracStr := parts.getD 1 ""
    let fracPart := fracStr.toNat?.getD 0
    let fracLen := fracStr.length
    pure <| intPart.toFloat + fracPart.toFloat / (10.0 ^ fracLen.toFloat)
  else
    match v.toNat? with
    | some n => pure n.toFloat
    | none => throw <| IO.userError s!"Invalid {name}: {v}"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest => parseArgsLoop rest { acc with source := v }
  | "--model-dir" :: v :: rest => parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest => parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest => parseArgsLoop rest { acc with cacheDir := v }
  | "--language" :: v :: rest => parseArgsLoop rest { acc with language := some v }
  | "--context" :: v :: rest => parseArgsLoop rest { acc with context := v }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--chunk-sec" :: v :: rest =>
      parseArgsLoop rest { acc with chunkSec := (← parseFloatArg "--chunk-sec" v) }
  | "--hop-sec" :: v :: rest =>
      parseArgsLoop rest { acc with hopSec := (← parseFloatArg "--hop-sec" v) }
  | "--run-sec" :: v :: rest =>
      parseArgsLoop rest { acc with runSec := (← parseFloatArg "--run-sec" v) }
  | "--simple-output" :: rest =>
      parseArgsLoop rest { acc with simpleOutput := true }
  | "--help" :: _ =>
      IO.println "Usage: lake exe Qwen3ASRLiveMic [options]"
      IO.println "  --source <path-or-repo>  Local model dir or HF repo id"
      IO.println "  --model-dir <path>       Alias for --source (backward compatible)"
      IO.println "  --revision <rev>         HF revision/branch/tag (default: main)"
      IO.println "  --cache-dir <path>       Local cache for downloaded files"
      IO.println "  --language <name>        Optional forced language"
      IO.println "  --context <text>         Optional system context"
      IO.println "  --max-new-tokens <n>     Greedy decode max new tokens"
      IO.println "  --chunk-sec <f>          Decode window seconds (overlap window)"
      IO.println "  --hop-sec <f>            Step seconds between decodes"
      IO.println "  --run-sec <f>            Total streaming duration"
      IO.println "  --simple-output          Print single transcript updates (no dual-track lines)"
      throw <| IO.userError ""
  | x :: _ => throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

private def toSamples (sec : Float) : Nat :=
  let n := ((sec * 16000.0) + 0.5).toUInt64.toNat
  if n == 0 then 1 else n

private def tailSlice (xs : Array Float) (n : Nat) : Array Float :=
  if xs.size <= n then xs else xs.extract (xs.size - n) xs.size

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if args.chunkSec <= 0.0 || args.hopSec <= 0.0 || args.runSec <= 0.0 then
    throw <| IO.userError "chunk-sec, hop-sec, and run-sec must be > 0"

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

  let chunkSamples := toSamples args.chunkSec
  let hopSamples := toSamples args.hopSec
  let steps := Nat.max 1 (((args.runSec / args.hopSec) + 0.5).toUInt64.toNat)

  IO.println s!"live mic: chunk_sec={args.chunkSec} hop_sec={args.hopSec} run_sec={args.runSec}"
  Tyr.Audio.AppleInput.start 16000 1 100

  try
    let mut ring : Array Float := #[]
    let mut tstate : RealtimeTranscriptState := { rollbackChars := 12 }
    let mut prevSimple := ""
    for _ in [:steps] do
      let pcm ← Tyr.Audio.AppleInput.read hopSamples.toUInt64 1500
      if !pcm.isEmpty then
        ring := ring ++ pcm
      if ring.size >= chunkSamples then
        let window := tailSlice ring chunkSamples
        let (tstate', delta) ← decodeOverlapStep
          model
          tok
          pre
          window
          tstate
          (context := args.context)
          (language := args.language)
          (maxNewTokens := args.maxNewTokens)
        tstate := tstate'
        if args.simpleOutput then
          if delta.fullText != prevSimple then
            IO.println s!"TEXT: {delta.fullText}"
            prevSimple := delta.fullText
        else
          if !delta.stableAppend.isEmpty then
            IO.println s!"STABLE+= {delta.stableAppend}"
          if !delta.unstableText.isEmpty then
            IO.println s!"UNSTABLE= {delta.unstableText}"
    Tyr.Audio.AppleInput.stop
  catch e =>
    Tyr.Audio.AppleInput.stop
    throw e

  pure 0

end Examples.Qwen3ASR

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen3ASR.runMain argv
