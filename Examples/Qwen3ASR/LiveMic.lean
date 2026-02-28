import Tyr.Model.Qwen3ASR
import Tyr.Tokenizer.Qwen3
import Tyr.Audio.AppleInput

namespace Examples.Qwen3ASR

open torch.qwen3asr

structure Args where
  modelDir : String := "weights/qwen3-asr-0.6b"
  language : Option String := none
  context : String := ""
  maxNewTokens : UInt64 := 128
  chunkSec : Float := 2.0
  hopSec : Float := 0.5
  runSec : Float := 30.0
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
  | "--model-dir" :: v :: rest => parseArgsLoop rest { acc with modelDir := v }
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
  | "--help" :: _ =>
      IO.println "Usage: lake exe Qwen3ASRLiveMic [options]"
      IO.println "  --model-dir <path>       Qwen3-ASR model directory"
      IO.println "  --language <name>        Optional forced language"
      IO.println "  --context <text>         Optional system context"
      IO.println "  --max-new-tokens <n>     Greedy decode max new tokens"
      IO.println "  --chunk-sec <f>          Decode window seconds (overlap window)"
      IO.println "  --hop-sec <f>            Step seconds between decodes"
      IO.println "  --run-sec <f>            Total streaming duration"
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

  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir args.modelDir
  let tok ← tokenizer.qwen3.loadTokenizer args.modelDir
  let pre ← PreprocessorConfig.loadFromPretrainedDir args.modelDir
  let model ← Qwen3ASRForConditionalGeneration.loadSharded args.modelDir cfg

  let chunkSamples := toSamples args.chunkSec
  let hopSamples := toSamples args.hopSec
  let steps := Nat.max 1 (((args.runSec / args.hopSec) + 0.5).toUInt64.toNat)

  IO.println s!"live mic: chunk_sec={args.chunkSec} hop_sec={args.hopSec} run_sec={args.runSec}"
  Tyr.Audio.AppleInput.start 16000 1 100

  try
    let mut ring : Array Float := #[]
    let mut tstate : RealtimeTranscriptState := { rollbackChars := 12 }
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
