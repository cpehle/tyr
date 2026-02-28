/-  
  Examples/Qwen35/RunHF.lean

  Text generation demo for Qwen3.5 models resolved by local path or
  HuggingFace repo id.
-/
import Tyr.Model.Qwen35
import Tyr.Model.Qwen35.Media
import Tyr.Tokenizer.Qwen3

open torch
open torch.qwen35

namespace Examples.Qwen35

structure Args where
  source : String := "tiny-random/qwen3.5"
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  prompt : String := "Give a concise definition of a dependent type."
  promptFile : Option String := none
  imagePath : Option String := none
  videoPath : Option String := none
  videoMaxFrames : UInt64 := 64
  batchSize : UInt64 := 1
  maxNewTokens : UInt64 := 32
  stream : Bool := false
  multimodal : Bool := false
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 := do
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def printHelp : IO Unit := do
  IO.println "Usage: lake exe Qwen35RunHF [options]"
  IO.println "  --source <path-or-repo>      Local model dir or HF repo id (default: tiny-random/qwen3.5)"
  IO.println "  --revision <rev>             HF revision/branch/tag (default: main)"
  IO.println "  --cache-dir <path>           Local cache for downloaded files"
  IO.println "  --prompt <text>              Prompt text"
  IO.println "  --prompt-file <path>         One prompt per non-empty line"
  IO.println "  --image <path>               Image file for multimodal generation (Apple-only)"
  IO.println "  --video <path>               Video file for multimodal generation (Apple-only)"
  IO.println "  --video-max-frames <n>       Max decoded video frames (default: 64)"
  IO.println "  --batch-size <n>             Prompts per decode batch (default: 1)"
  IO.println "  --max-new-tokens <n>         Number of tokens to generate"
  IO.println "  --stream                     Stream generated tokens per decode step"
  IO.println "  --multimodal                 Load Qwen35ForConditionalGeneration"
  IO.println "Examples:"
  IO.println "  lake exe Qwen35RunHF --source tiny-random/qwen3.5 --stream"
  IO.println "  lake exe Qwen35RunHF --source Qwen/Qwen3.5-4B --prompt \"Write one sentence about Lean.\""
  IO.println "  lake exe Qwen35RunHF --source tiny-random/qwen3.5 --prompt-file prompts.txt --batch-size 4"

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
  | "--image" :: v :: rest =>
      parseArgsLoop rest { acc with imagePath := some v }
  | "--video" :: v :: rest =>
      parseArgsLoop rest { acc with videoPath := some v }
  | "--video-max-frames" :: v :: rest =>
      parseArgsLoop rest { acc with videoMaxFrames := (← parseNatArg "--video-max-frames" v) }
  | "--batch-size" :: v :: rest =>
      parseArgsLoop rest { acc with batchSize := (← parseNatArg "--batch-size" v) }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--stream" :: rest =>
      parseArgsLoop rest { acc with stream := true }
  | "--multimodal" :: rest =>
      parseArgsLoop rest { acc with multimodal := true }
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

private def visionPrefixIds (cfg : VLConfig) (imageTokenCount videoTokenCount : UInt64) : Array UInt64 :=
  Id.run do
    let mut out : Array UInt64 := #[]
    if imageTokenCount > 0 then
      out := out.push cfg.vision_start_token_id
      for _ in [:imageTokenCount.toNat] do
        out := out.push cfg.image_token_id
      out := out.push cfg.vision_end_token_id
    if videoTokenCount > 0 then
      out := out.push cfg.vision_start_token_id
      for _ in [:videoTokenCount.toNat] do
        out := out.push cfg.video_token_id
      out := out.push cfg.vision_end_token_id
    out

private def encodePromptToIdsMultimodal
    (tok : tokenizer.qwen3.QwenTokenizer)
    (cfg : VLConfig)
    (imageTokenCount videoTokenCount : UInt64)
    (prompt : String)
    : Array UInt64 :=
  (visionPrefixIds cfg imageTokenCount videoTokenCount) ++ (encodePromptToIds tok prompt)

private def buildBatchInputWithEncoder
    (tok : tokenizer.qwen3.QwenTokenizer)
    (prompts : Array String)
    (encode : String → Array UInt64)
    : IO (Sigma (fun batch => Sigma (fun seq => T #[batch, seq] × Array Nat))) := do
  let encoded := prompts.map encode
  let batch := encoded.size.toUInt64
  if batch == 0 then
    throw <| IO.userError "buildBatchInputWithEncoder requires at least one prompt"

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

private def eosFromCfg (cfg : Config) : Array UInt64 :=
  match cfg.eos_token_id with
  | some id => #[id]
  | none => #[]

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

private def streamCallback
    (tok : tokenizer.qwen3.QwenTokenizer)
    {batch : UInt64}
    (chunkStart : Nat)
    : Qwen35ForCausalLM.StreamCallback batch := fun _step nextTok => do
  let flat : T #[batch] := reshape (data.toLong nextTok) #[batch]
  let vals ← data.tensorToUInt64Array flat
  if batch == 1 then
    match vals[0]? with
    | some v =>
      let piece := tokenizer.qwen3.decodeOne tok v.toUInt32
      IO.print piece
    | none => pure ()
  else
    for i in [:vals.size] do
      let idx := chunkStart + i
      let piece := tokenizer.qwen3.decodeOne tok vals[i]!.toUInt32
      IO.println s!"STREAM[{idx}] {piece}"

private def runTextBatches
    (tok : tokenizer.qwen3.QwenTokenizer)
    (cfg : Config)
    (model : Qwen35ForCausalLM cfg)
    (args : Args)
    (prompts : Array String)
    : IO Unit := do
  let chunkSize := Nat.max 1 args.batchSize.toNat
  let eos := eosFromCfg cfg

  let mut start : Nat := 0
  while start < prompts.size do
    let stop := Nat.min prompts.size (start + chunkSize)
    let chunk := prompts.extract start stop
    let ⟨_batch, ⟨_seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok chunk (encodePromptToIds tok)

    let ⟨_outSeq, outIds⟩ ←
      if args.stream then
        model.generateStream
          cfg
          inputIds
          (streamCallback tok start)
          args.maxNewTokens
          .greedy
          eos
      else
        model.generate cfg inputIds args.maxNewTokens .greedy eos

    if args.stream && chunk.size == 1 then
      IO.println ""
    let decoded ← decodeGeneratedBatch tok promptLens outIds
    printDecodedBatch start decoded (singleOnly := prompts.size == 1)
    start := stop

private def runMultimodalBatches
    (tok : tokenizer.qwen3.QwenTokenizer)
    (cfg : VLConfig)
    (model : Qwen35ForConditionalGeneration cfg)
    (args : Args)
    (prompts : Array String)
    (imageTokenCount : UInt64)
    (videoTokenCount : UInt64)
    (imageFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])))
    (videoFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])))
    : IO Unit := do
  let chunkSize := Nat.max 1 args.batchSize.toNat
  let eos := eosFromCfg cfg.text_config

  let mut start : Nat := 0
  while start < prompts.size do
    let stop := Nat.min prompts.size (start + chunkSize)
    let chunk := prompts.extract start stop
    let enc := encodePromptToIdsMultimodal tok cfg imageTokenCount videoTokenCount
    let ⟨_batch, ⟨_seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok chunk enc

    let ⟨_outSeq, outIds⟩ ←
      if args.stream then
        model.generateStream
          cfg
          inputIds
          (streamCallback tok start)
          args.maxNewTokens
          .greedy
          eos
          imageFeatures
          videoFeatures
      else
        model.generate
          cfg
          inputIds
          args.maxNewTokens
          .greedy
          eos
          imageFeatures
          videoFeatures

    if args.stream && chunk.size == 1 then
      IO.println ""
    let decoded ← decodeGeneratedBatch tok promptLens outIds
    printDecodedBatch start decoded (singleOnly := prompts.size == 1)
    start := stop

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  let modelDir ← hub.resolvePretrainedDir args.source {
    revision := args.revision
    cacheDir := args.cacheDir
    includeTokenizer := true
  }
  IO.println s!"Model directory: {modelDir}"

  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let prompts ← loadPrompts args

  if args.multimodal then
    let cfg ← VLConfig.loadFromPretrainedDir modelDir {}
    let isSharded ← hub.detectWeightLayout modelDir
    let model ←
      if isSharded then
        Qwen35ForConditionalGeneration.loadSharded modelDir cfg
      else
        Qwen35ForConditionalGeneration.load s!"{modelDir}/model.safetensors" cfg
    let imagePatches? ←
      match args.imagePath with
      | some p =>
        IO.println s!"Loading image patches from {p}..."
        pure (some (← media.loadImagePatches cfg p))
      | none => pure none
    let videoPatches? ←
      match args.videoPath with
      | some p =>
        IO.println s!"Loading video patches from {p} (maxFrames={args.videoMaxFrames})..."
        pure (some (← media.loadVideoPatches cfg p args.videoMaxFrames))
      | none => pure none

    let imageFeatures? ←
      match imagePatches? with
      | some ⟨nPatches, patches⟩ =>
        let feats ← model.getImageFeatures cfg patches
        let nTok := VisionConfig.mergedTokenCount cfg.vision_config nPatches
        pure (some ⟨nTok, feats⟩)
      | none => pure none
    let videoFeatures? ←
      match videoPatches? with
      | some ⟨nPatches, patches⟩ =>
        let feats ← model.getVideoFeatures cfg patches
        let nTok := VisionConfig.mergedTokenCount cfg.vision_config nPatches
        pure (some ⟨nTok, feats⟩)
      | none => pure none

    let imageTokenCount :=
      match imageFeatures? with
      | some ⟨n, _⟩ => n
      | none => 0
    let videoTokenCount :=
      match videoFeatures? with
      | some ⟨n, _⟩ => n
      | none => 0

    if imageTokenCount == 0 && videoTokenCount == 0 then
      IO.println "Warning: --multimodal enabled but no --image/--video provided; running text-only path."

    runMultimodalBatches
      tok cfg model args prompts
      imageTokenCount videoTokenCount
      imageFeatures? videoFeatures?
  else
    let cfg ← Config.loadFromPretrainedDir modelDir Config.qwen35_9B
    let isSharded ← hub.detectWeightLayout modelDir
    let model ←
      if isSharded then
        Qwen35ForCausalLM.loadSharded modelDir cfg
      else
        Qwen35ForCausalLM.load s!"{modelDir}/model.safetensors" cfg
    runTextBatches tok cfg model args prompts

  pure 0

end Examples.Qwen35

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen35.runMain argv
