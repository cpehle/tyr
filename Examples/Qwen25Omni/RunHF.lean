/-  
  Examples/Qwen25Omni/RunHF.lean

  Text generation demo for Qwen2.5-Omni thinker checkpoints (3B/7B),
  resolved by local path or HuggingFace repo id.
-/
import Tyr.Model.Qwen25Omni
import Tyr.Tokenizer.Qwen3

open torch
open torch.qwen25omni

namespace Examples.Qwen25Omni

structure Args where
  source : String := "Qwen/Qwen2.5-Omni-3B"
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  prompt : String := "Give a concise definition of a dependent type."
  promptFile : Option String := none
  batchSize : UInt64 := 1
  maxNewTokens : UInt64 := 64
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 := do
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def printHelp : IO Unit := do
  IO.println "Usage: lake exe Qwen25OmniRunHF [options]"
  IO.println "  --source <path-or-repo>      Local model dir or HF repo id (default: Qwen/Qwen2.5-Omni-3B)"
  IO.println "  --revision <rev>             HF revision/branch/tag (default: main)"
  IO.println "  --cache-dir <path>           Local cache for downloaded files"
  IO.println "  --prompt <text>              Prompt text"
  IO.println "  --prompt-file <path>         One prompt per non-empty line"
  IO.println "  --batch-size <n>             Prompts per decode batch (default: 1)"
  IO.println "  --max-new-tokens <n>         Number of tokens to generate"
  IO.println "Examples:"
  IO.println "  lake exe Qwen25OmniRunHF --source Qwen/Qwen2.5-Omni-3B --prompt \"Hello\""
  IO.println "  lake exe Qwen25OmniRunHF --source Qwen/Qwen2.5-Omni-7B --prompt-file prompts.txt --batch-size 2"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest =>
      parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest =>
      parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest =>
      parseArgsLoop rest { acc with cacheDir := v }
  | "--prompt" :: v :: rest =>
      parseArgsLoop rest { acc with prompt := v }
  | "--prompt-file" :: v :: rest =>
      parseArgsLoop rest { acc with promptFile := some v }
  | "--batch-size" :: v :: rest =>
      parseArgsLoop rest { acc with batchSize := (← parseNatArg "--batch-size" v) }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--help" :: _ =>
      printHelp
      throw <| IO.userError ""
  | x :: _ =>
      throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

private def loadPrompts (args : Args) : IO (Array String) := do
  match args.promptFile with
  | some path =>
    let contents ← IO.FS.readFile path
    let lines := contents.splitOn "\n"
    let prompts := lines.foldl
      (init := #[])
      (fun acc line =>
        let s := line.trim
        if s.isEmpty then acc else acc.push s)
    if prompts.isEmpty then
      throw <| IO.userError s!"No prompts found in {path}"
    pure prompts
  | none =>
    pure #[args.prompt]

private def encodePromptToIds
    (tok : tokenizer.qwen3.QwenTokenizer)
    (prompt : String)
    : Array UInt64 :=
  let text := tokenizer.qwen3.chatTemplate prompt
  (tokenizer.qwen3.encodeText tok text).map (fun t => t.toUInt64)

private def buildBatchInput
    (tok : tokenizer.qwen3.QwenTokenizer)
    (prompts : Array String)
    : IO (Sigma (fun batch => Sigma (fun seq => T #[batch, seq] × Array Nat))) := do
  let encoded := prompts.map (encodePromptToIds tok)
  let batch := encoded.size.toUInt64
  if batch == 0 then
    throw <| IO.userError "buildBatchInput requires at least one prompt"

  let maxLenNat := encoded.foldl (fun m ids => Nat.max m ids.size) 0
  if maxLenNat == 0 then
    throw <| IO.userError "Prompt tokenization produced empty input."
  let seq := maxLenNat.toUInt64

  let mut flat : Array Int64 := #[]
  let mut promptLens : Array Nat := #[]
  for ids in encoded do
    promptLens := promptLens.push ids.size
    let mut row : Array Int64 := ids.map (fun x => x.toInt64)
    while row.size < maxLenNat do
      row := row.push tok.padToken.toUInt64.toInt64
    flat := flat ++ row

  let inputIds : T #[batch, seq] := reshape (data.fromInt64Array flat) #[batch, seq]
  pure ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩

private def decodeGeneratedBatch
    (tok : tokenizer.qwen3.QwenTokenizer)
    (promptLens : Array Nat)
    {batch outSeq : UInt64}
    (ids : T #[batch, outSeq])
    : IO (Array String) := do
  let mut out : Array String := #[]
  for i in [:batch.toNat] do
    let row2 : T #[1, outSeq] := data.slice ids 0 i.toUInt64 1
    let row1 : T #[outSeq] := reshape (data.toLong row2) #[outSeq]
    let vals ← data.tensorToUInt64Array row1
    let promptLen := promptLens.getD i 0
    let gen :=
      if vals.size <= promptLen then
        #[]
      else
        vals.extract promptLen vals.size
    let text := tokenizer.qwen3.decodeText tok (gen.map (fun x => x.toUInt32))
    out := out.push text
  pure out

private def printDecodedBatch
    (chunkStart : Nat)
    (decoded : Array String)
    (singleOnly : Bool := false)
    : IO Unit := do
  if singleOnly && decoded.size == 1 && chunkStart == 0 then
    IO.println "GEN_BEGIN"
    IO.println decoded[0]!
    IO.println "GEN_END"
  else
    for i in [:decoded.size] do
      let idx := chunkStart + i
      IO.println s!"GEN[{idx}]_BEGIN"
      IO.println decoded[i]!
      IO.println s!"GEN[{idx}]_END"

private def defaultConfigForSource (source : String) : Config :=
  if source.contains "-7B" then
    Config.qwen25omni_7B
  else
    Config.qwen25omni_3B

private def runBatches
    (tok : tokenizer.qwen3.QwenTokenizer)
    (model : Qwen25OmniForCausalLM cfg)
    (args : Args)
    (prompts : Array String)
    : IO Unit := do
  let chunkSize := Nat.max 1 args.batchSize.toNat
  let eos : Array UInt64 := #[]

  let mut start : Nat := 0
  while start < prompts.size do
    let stop := Nat.min prompts.size (start + chunkSize)
    let chunk := prompts.extract start stop
    let ⟨_batch, ⟨_seq, (inputIds, promptLens)⟩⟩ ← buildBatchInput tok chunk
    let ⟨_outSeq, outIds⟩ ← model.generateGreedy inputIds args.maxNewTokens eos
    let decoded ← decodeGeneratedBatch tok promptLens outIds
    printDecodedBatch start decoded (singleOnly := prompts.size == 1)
    start := stop

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  let prompts ← loadPrompts args

  let modelDir ← hub.resolvePretrainedDir args.source {
    revision := args.revision
    cacheDir := args.cacheDir
    includeTokenizer := true
  }
  IO.println s!"Resolved model dir: {modelDir}"

  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let cfgDefaults := defaultConfigForSource args.source
  let cfg ← Config.loadFromPretrainedDir modelDir cfgDefaults
  let isSharded ← hub.detectWeightLayout modelDir
  let model ←
    if isSharded then
      Qwen25OmniForCausalLM.loadSharded modelDir cfg
    else
      Qwen25OmniForCausalLM.load s!"{modelDir}/model.safetensors" cfg

  runBatches tok model args prompts
  pure 0

end Examples.Qwen25Omni

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen25Omni.runMain argv
