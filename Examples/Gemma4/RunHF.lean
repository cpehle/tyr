/-
  Examples/Gemma4/RunHF.lean

  Text and image-conditioned generation demo for Gemma 4 models resolved by
  local path or HuggingFace repo id.
-/
import Tyr.Model.Gemma4
import Tyr.Tokenizer.Gemma4

open torch
open torch.gemma4

namespace Examples.Gemma4

structure Args where
  source : String := "google/gemma-4-E4B"
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  device : String := "auto"
  prompt : String := "Give a concise definition of a dependent type."
  promptFile : Option String := none
  imagePaths : Array String := #[]
  batchSize : UInt64 := 1
  maxNewTokens : UInt64 := 32
  stream : Bool := false
  multimodal : Bool := false
  enableThinking : Bool := false
  debugIds : Bool := false
  showHelp : Bool := false
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 := do
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def printHelp : IO Unit := do
  IO.println "Usage: lake exe Gemma4RunHF [options]"
  IO.println "  --source <path-or-repo>      Local model dir or HF repo id (default: google/gemma-4-E4B)"
  IO.println "  --revision <rev>             HF revision/branch/tag (default: main)"
  IO.println "  --cache-dir <path>           Local cache for downloaded files"
  IO.println "  --device <auto|cpu|mps|cuda[:n]>  Execution device (default: auto)"
  IO.println "  --prompt <text>              Prompt text"
  IO.println "  --prompt-file <path>         One prompt per non-empty line"
  IO.println "  --image <path>               Image file for multimodal generation (repeatable, Apple-only)"
  IO.println "  --batch-size <n>             Prompts per decode batch (default: 1)"
  IO.println "  --max-new-tokens <n>         Number of tokens to generate"
  IO.println "  --stream                     Stream generated tokens per decode step"
  IO.println "  --multimodal                 Force multimodal model load (auto-enabled by --image)"
  IO.println "  --enable-thinking            Use the Gemma 4 thinking-enabled chat template"
  IO.println "  --debug-ids                  Print generated token ids alongside decoded text"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest =>
      parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest =>
      parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest =>
      parseArgsLoop rest { acc with cacheDir := v }
  | "--device" :: v :: rest =>
      parseArgsLoop rest { acc with device := v }
  | "--prompt" :: v :: rest =>
      parseArgsLoop rest { acc with prompt := v }
  | "--prompt-file" :: v :: rest =>
      parseArgsLoop rest { acc with promptFile := some v }
  | "--image" :: v :: rest =>
      parseArgsLoop rest { acc with imagePaths := acc.imagePaths.push v }
  | "--batch-size" :: v :: rest =>
      parseArgsLoop rest { acc with batchSize := (← parseNatArg "--batch-size" v) }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--stream" :: rest =>
      parseArgsLoop rest { acc with stream := true }
  | "--multimodal" :: rest =>
      parseArgsLoop rest { acc with multimodal := true }
  | "--enable-thinking" :: rest =>
      parseArgsLoop rest { acc with enableThinking := true }
  | "--debug-ids" :: rest =>
      parseArgsLoop rest { acc with debugIds := true }
  | "--help" :: _ =>
      parseArgsLoop [] { acc with showHelp := true }
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
        let s := line.trimAscii.toString
        if s.isEmpty then acc else acc.push s)
    if prompts.isEmpty then
      throw <| IO.userError s!"No prompts found in {path}"
    pure prompts
  | none =>
    pure #[args.prompt]

private def deviceToString : Device → String
  | Device.MPS => "MPS"
  | Device.CPU => "CPU"
  | Device.CUDA n => s!"CUDA:{n}"

private def resolveDevice (arg : String) : IO Device := do
  let requested := arg.trimAscii.toString.toLower
  match requested with
  | "auto" => getBestDevice
  | "cpu" => pure Device.CPU
  | "mps" => pure Device.MPS
  | "cuda" =>
      if ← cuda_is_available then pure (Device.CUDA 0) else pure Device.CPU
  | _ =>
      if requested.startsWith "cuda:" then
        match (requested.drop 5).toNat? with
        | some idx =>
            if ← cuda_is_available then pure (Device.CUDA idx.toUInt64) else pure Device.CPU
        | none => pure Device.CPU
      else
        pure Device.CPU

private def moveModelToDevice [TensorStruct α] (device : Device) (x : α) : IO α :=
  TensorStruct.mapM (fun t => pure (t.to device)) x

private def movePatchGridTo {cfg : VLConfig}
    (device : Device)
    (x : ImagePatchGrid cfg)
    : ImagePatchGrid cfg :=
  match x with
  | ⟨patchRows, ⟨patchCols, grid⟩⟩ =>
    ⟨patchRows, ⟨patchCols, grid.to device⟩⟩

private def encodePromptToIds
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (enableThinking : Bool)
    (prompt : String)
    : Array UInt64 :=
  let text :=
    if enableThinking then
      tokenizer.gemma4.chatTemplateThinking prompt
    else
      tokenizer.gemma4.chatTemplate prompt
  (tokenizer.gemma4.encodeText tok text).map (fun t => t.toUInt64)

private def buildImageExpansion (softTokenCount : UInt64) : String :=
  Id.run do
    let mut out := "<|image>"
    for _ in [:softTokenCount.toNat] do
      out := out ++ "<|image|>"
    out := out ++ "<image|>"
    pure out

private def prefixImageExpansions (softTokenCounts : Array UInt64) : String :=
  Id.run do
    let mut out := ""
    for i in [:softTokenCounts.size] do
      if i > 0 then
        out := out ++ " "
      out := out ++ buildImageExpansion softTokenCounts[i]!
    pure out

private def injectImageExpansions
    (prompt : String)
    (softTokenCounts : Array UInt64)
    : IO String := do
  let placeholder := "<|image|>"
  let parts := (prompt.splitOn placeholder).toArray
  if parts.size <= 1 then do
    let prefixText := prefixImageExpansions softTokenCounts
    if prefixText.isEmpty then
      pure prompt
    else if prompt.trimAscii.isEmpty then
      pure prefixText
    else
      pure s!"{prefixText}\n{prompt}"
  else do
    let occurrences := parts.size - 1
    if occurrences != softTokenCounts.size then
      throw <| IO.userError
        s!"Prompt contains {occurrences} image placeholders but {softTokenCounts.size} image inputs were provided"
    let mut out := parts[0]!
    for i in [:occurrences] do
      out := out ++ buildImageExpansion softTokenCounts[i]! ++ parts[i + 1]!
    pure out

private def encodePromptToIdsMultimodal
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (enableThinking : Bool)
    (softTokenCounts : Array UInt64)
    (prompt : String)
    : IO (Array UInt64) := do
  let prompt' ← injectImageExpansions prompt softTokenCounts
  pure (encodePromptToIds tok enableThinking prompt')

private def buildBatchInputWithEncoder
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (prompts : Array String)
    (encode : String → IO (Array UInt64))
    : IO (Sigma (fun batch => Sigma (fun seq => T #[batch, seq] × Array Nat))) := do
  let mut encoded : Array (Array UInt64) := #[]
  for prompt in prompts do
    encoded := encoded.push (← encode prompt)
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

private def pushUnique (xs : Array UInt64) (x : UInt64) : Array UInt64 :=
  if xs.contains x then xs else xs.push x

private def eosStopTokenIds (tok : tokenizer.gemma4.GemmaTokenizer) (cfg : Config) : Array UInt64 :=
  Id.run do
    let mut out : Array UInt64 := #[]
    match cfg.eos_token_id with
    | some id => out := pushUnique out id
    | none =>
      match tok.eosToken with
      | some id => out := pushUnique out id.toUInt64
      | none => pure ()
    out

private def generatedIdsFromBatch
    (promptLens : Array Nat)
    {batch outSeq : UInt64}
    (ids : T #[batch, outSeq])
    : IO (Array (Array UInt64)) := do
  let mut out : Array (Array UInt64) := #[]
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
    out := out.push gen
  pure out

private def decodeGeneratedBatch
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (promptLens : Array Nat)
    {batch outSeq : UInt64}
    (ids : T #[batch, outSeq])
    : IO (Array String) := do
  let generatedIds ← generatedIdsFromBatch promptLens ids
  pure <| generatedIds.map (fun xs => tokenizer.gemma4.decodeText tok (xs.map (fun x => x.toUInt32)))

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

private def printGeneratedIds
    (chunkStart : Nat)
    (generatedIds : Array (Array UInt64))
    (singleOnly : Bool := false)
    : IO Unit := do
  if singleOnly && generatedIds.size == 1 && chunkStart == 0 then
    IO.println s!"GEN_IDS={generatedIds[0]!}"
  else
    for i in [:generatedIds.size] do
      let idx := chunkStart + i
      IO.println s!"GEN_IDS[{idx}]={generatedIds[i]!}"

private def streamCallback
    (tok : tokenizer.gemma4.GemmaTokenizer)
    {batch : UInt64}
    (chunkStart : Nat)
    : Gemma4ForCausalLM.StreamCallback batch := fun _step nextTok => do
  let flat : T #[batch] := reshape (data.toLong nextTok) #[batch]
  let vals ← data.tensorToUInt64Array flat
  if batch == 1 then
    match vals[0]? with
    | some v =>
      let piece := tokenizer.gemma4.decodeOne tok v.toUInt32
      IO.print piece
    | none => pure ()
  else
    for i in [:vals.size] do
      let idx := chunkStart + i
      let piece := tokenizer.gemma4.decodeOne tok vals[i]!.toUInt32
      IO.println s!"STREAM[{idx}] {piece}"

private def runTextBatches
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (cfg : Config)
    (model : Gemma4ForCausalLM cfg)
    (device : Device)
    (args : Args)
    (prompts : Array String)
    : IO Unit := do
  let chunkSize := Nat.max 1 args.batchSize.toNat
  let generationStopIds := eosStopTokenIds tok cfg

  let mut start : Nat := 0
  while start < prompts.size do
    let stop := Nat.min prompts.size (start + chunkSize)
    let chunk := prompts.extract start stop
    let ⟨_batch, ⟨_seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok chunk (fun p => pure (encodePromptToIds tok args.enableThinking p))
    let inputIds := inputIds.to device

    let ⟨_outSeq, outIds⟩ ←
      if args.stream then
        model.generateStream
          cfg
          inputIds
          (streamCallback tok start)
          args.maxNewTokens
          .greedy
          generationStopIds
      else
        model.generate cfg inputIds args.maxNewTokens .greedy generationStopIds

    if args.stream && chunk.size == 1 then
      IO.println ""

    if !args.stream then
      let decoded ← decodeGeneratedBatch tok promptLens outIds
      if args.debugIds then
        let generatedIds ← generatedIdsFromBatch promptLens outIds
        printGeneratedIds start generatedIds (singleOnly := prompts.size == 1)
      printDecodedBatch start decoded (singleOnly := prompts.size == 1)
    start := stop

private def runMultimodalBatches
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (cfg : VLConfig)
    (model : Gemma4ForConditionalGeneration cfg)
    (device : Device)
    (args : Args)
    (prompts : Array String)
    (imageSoftTokenCounts : Array UInt64)
    (imageFeatures : Option (ImageFeatures cfg))
    : IO Unit := do
  let chunkSize := Nat.max 1 args.batchSize.toNat
  let generationStopIds := eosStopTokenIds tok cfg.text_config

  let mut start : Nat := 0
  while start < prompts.size do
    let stop := Nat.min prompts.size (start + chunkSize)
    let chunk := prompts.extract start stop
    let enc := encodePromptToIdsMultimodal tok args.enableThinking imageSoftTokenCounts
    let ⟨batch, ⟨_seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok chunk enc
    let inputIds := inputIds.to device
    let imageFeaturesChunk :=
      match imageFeatures with
      | some ⟨_nTok, feats⟩ =>
        some <| Gemma4ForConditionalGeneration.repeatFeaturesForBatch (batch := batch) cfg feats
      | none =>
        none

    let ⟨_outSeq, outIds⟩ ←
      if args.stream then
        model.generateStream
          cfg
          inputIds
          (streamCallback tok start)
          args.maxNewTokens
          .greedy
          generationStopIds
          imageFeaturesChunk
      else
        model.generate
          cfg
          inputIds
          args.maxNewTokens
          .greedy
          generationStopIds
          imageFeaturesChunk

    if args.stream && chunk.size == 1 then
      IO.println ""

    if !args.stream then
      let decoded ← decodeGeneratedBatch tok promptLens outIds
      if args.debugIds then
        let generatedIds ← generatedIdsFromBatch promptLens outIds
        printGeneratedIds start generatedIds (singleOnly := prompts.size == 1)
      printDecodedBatch start decoded (singleOnly := prompts.size == 1)
    start := stop

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if args.showHelp then
    printHelp
    return 0

  let device ← resolveDevice args.device
  let hasImages := !args.imagePaths.isEmpty
  let useMultimodal := args.multimodal || hasImages
  if hasImages && !args.multimodal then
    IO.println "Info: enabling multimodal mode because --image was provided."

  let modelDir ← hub.resolvePretrainedDir args.source {
    revision := args.revision
    cacheDir := args.cacheDir
    includeTokenizer := true
  }
  IO.println s!"Model directory: {modelDir}"
  IO.println s!"Using device: {deviceToString device}"

  let tok ← tokenizer.gemma4.loadTokenizer modelDir
  let prompts ← loadPrompts args

  if useMultimodal && hasImages then
    let cfg ← VLConfig.loadFromPretrainedDir modelDir {}
    let isSharded ← hub.detectWeightLayout modelDir
    let modelCpu ←
      if isSharded then
        Gemma4ForConditionalGeneration.loadSharded modelDir cfg
      else
        Gemma4ForConditionalGeneration.load s!"{modelDir}/model.safetensors" cfg
    let model ← moveModelToDevice device modelCpu

    let mut imageGrids : Array (ImagePatchGrid cfg) := #[]
    let mut imageSoftTokenCounts : Array UInt64 := #[]
    for p in args.imagePaths do
      IO.println s!"Loading image patches from {p}..."
      let grid := movePatchGridTo device (← media.loadImagePatchGrid cfg p)
      match grid with
      | ⟨patchRows, ⟨patchCols, _⟩⟩ =>
        imageSoftTokenCounts :=
          imageSoftTokenCounts.push
            ((patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size))
      imageGrids := imageGrids.push grid

    let imageFeatures ← model.getImageFeaturesMany cfg imageGrids
    runMultimodalBatches tok cfg model device args prompts imageSoftTokenCounts imageFeatures
  else
    if useMultimodal && !hasImages then
      IO.println "Warning: --multimodal requested without --image; running text-only path."
    let cfg ← Config.loadFromPretrainedDir modelDir Config.gemma4_E4B
    let isSharded ← hub.detectWeightLayout modelDir
    let modelCpu ←
      if isSharded then
        Gemma4ForCausalLM.loadSharded modelDir cfg
      else
        Gemma4ForCausalLM.load s!"{modelDir}/model.safetensors" cfg
    let model ← moveModelToDevice device modelCpu
    runTextBatches tok cfg model device args prompts

  pure 0

end Examples.Gemma4

def main (argv : List String) : IO UInt32 :=
  Examples.Gemma4.runMain argv
