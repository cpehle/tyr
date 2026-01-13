/-
  Tyr/Eval/CORE.lean

  CORE metric evaluation framework based on nanochat's core_eval.py.

  The CORE metric (from DCLM paper: arxiv.org/abs/2406.11794) evaluates
  models on a standardized set of tasks using loss-based scoring.

  Supports three task types:
  - multiple_choice: Select option with lowest average continuation loss
  - schema: Context varies, continuation same (common suffix)
  - language_modeling: Predict continuation tokens exactly

  Key features:
  - Few-shot example rendering
  - Common prefix/suffix detection for efficient batching
  - Distributed evaluation support
-/
import Tyr.Torch
import Tyr.Distributed

namespace torch.Eval.CORE

open torch

/-! ## Task Types -/

/-- Types of evaluation tasks -/
inductive TaskType where
  | multipleChoice   : TaskType
  | schema           : TaskType
  | languageModeling : TaskType
  deriving Repr, BEq, Inhabited

/-- Task metadata for evaluation -/
structure TaskMeta where
  /-- Task type (MC, schema, LM) -/
  taskType : TaskType
  /-- Number of few-shot examples -/
  numFewshot : Nat := 0
  /-- Delimiter between context and continuation -/
  continuationDelimiter : String := ""
  /-- Task name for logging -/
  taskName : String := "unknown"
  deriving Repr, Inhabited

/-! ## Data Items -/

/-- Multiple choice item -/
structure MCItem where
  /-- Query/question text -/
  query : String
  /-- List of choice texts -/
  choices : Array String
  /-- Index of correct answer (0-indexed) -/
  gold : Nat
  deriving Repr, Inhabited

/-- Schema item (varying context, same continuation) -/
structure SchemaItem where
  /-- List of context options -/
  contextOptions : Array String
  /-- Common continuation -/
  continuation : String
  /-- Index of correct context (0-indexed) -/
  gold : Nat
  deriving Repr, Inhabited

/-- Language modeling item -/
structure LMItem where
  /-- Context/prefix text -/
  context : String
  /-- Continuation to predict -/
  continuation : String
  deriving Repr, Inhabited

/-- Generic evaluation item (union type) -/
inductive EvalItem where
  | mc : MCItem → EvalItem
  | schema : SchemaItem → EvalItem
  | lm : LMItem → EvalItem
  deriving Repr, Inhabited

/-! ## Prompt Rendering -/

/-- Render few-shot examples for multiple choice -/
def renderFewshotMC (examples : Array MCItem) (delimiter : String) : String := Id.run do
  let mut result := ""
  for ex in examples do
    let answer := ex.choices[ex.gold]!
    result := result ++ ex.query ++ delimiter ++ answer ++ "\n\n"
  result

/-- Render prompts for multiple choice question.
    Returns one prompt per choice option. -/
def renderPromptsMC (item : MCItem) (delimiter : String) (fewshot : Array MCItem) : Array String := Id.run do
  let fewshotStr := renderFewshotMC fewshot delimiter
  let mut prompts := #[]
  for choice in item.choices do
    prompts := prompts.push (fewshotStr ++ item.query ++ delimiter ++ choice)
  prompts

/-- Render few-shot examples for schema tasks -/
def renderFewshotSchema (examples : Array SchemaItem) (delimiter : String) : String := Id.run do
  let mut result := ""
  for ex in examples do
    let context := ex.contextOptions[ex.gold]!
    result := result ++ context ++ delimiter ++ ex.continuation ++ "\n\n"
  result

/-- Render prompts for schema question.
    Returns one prompt per context option. -/
def renderPromptsSchema (item : SchemaItem) (delimiter : String) (fewshot : Array SchemaItem)
    : Array String := Id.run do
  let fewshotStr := renderFewshotSchema fewshot delimiter
  let mut prompts := #[]
  for ctx in item.contextOptions do
    prompts := prompts.push (fewshotStr ++ ctx ++ delimiter ++ item.continuation)
  prompts

/-- Render few-shot examples for LM tasks -/
def renderFewshotLM (examples : Array LMItem) (delimiter : String) : String := Id.run do
  let mut result := ""
  for ex in examples do
    result := result ++ ex.context.trimAscii.toString ++ delimiter ++ ex.continuation ++ "\n\n"
  result

/-- Render prompts for language modeling task.
    Returns two prompts: without continuation (for finding prefix length) and with. -/
def renderPromptsLM (item : LMItem) (delimiter : String) (fewshot : Array LMItem)
    : (String × String) :=
  let fewshotStr := renderFewshotLM fewshot delimiter
  let promptWithout := fewshotStr ++ item.context.trimAscii.toString ++ delimiter
  let promptWith := promptWithout ++ item.continuation
  (promptWithout.trimAscii.toString, promptWith)

/-! ## Sequence Processing -/

/-- Find length of common prefix across token sequences -/
def findCommonPrefixLength (sequences : Array (Array UInt64)) : Nat := Id.run do
  if sequences.isEmpty then return 0
  let minLen := sequences.foldl (fun acc seq => min acc seq.size) sequences[0]!.size
  for i in [:minLen] do
    let firstToken := sequences[0]![i]!
    for seq in sequences do
      if seq[i]! != firstToken then
        return i
  minLen

/-- Find length of common suffix across token sequences -/
def findCommonSuffixLength (sequences : Array (Array UInt64)) : Nat := Id.run do
  if sequences.isEmpty then return 0
  let minLen := sequences.foldl (fun acc seq => min acc seq.size) sequences[0]!.size
  for i in [:minLen] do
    let idx := sequences[0]!.size - 1 - i
    let firstToken := sequences[0]![idx]!
    for seq in sequences do
      let seqIdx := seq.size - 1 - i
      if seq[seqIdx]! != firstToken then
        return i
  minLen

/-- Pad sequences to same length -/
def padSequences (sequences : Array (Array UInt64)) (padToken : UInt64) : Array (Array UInt64) := Id.run do
  if sequences.isEmpty then return #[]
  let maxLen := sequences.foldl (fun acc seq => max acc seq.size) 0
  let mut result := #[]
  for seq in sequences do
    let padding := (List.replicate (maxLen - seq.size) padToken).toArray
    result := result.push (seq ++ padding)
  result

/-- Convert padded sequences to input tensor -/
def sequencesToTensor {batch seq : UInt64} (sequences : Array (Array UInt64))
    : T #[batch, seq] :=
  -- Flatten and convert to int64 array
  let flat := sequences.foldl (fun acc seq => acc ++ seq.map (·.toInt64)) #[]
  let tensor := data.fromInt64Array flat
  reshape tensor #[batch, seq]

/-! ## Batching -/

/-- Batch info for evaluation -/
structure BatchInfo where
  /-- Token sequences (one per option/choice) -/
  tokens : Array (Array UInt64)
  /-- Start index of continuation for each sequence -/
  startIndices : Array Nat
  /-- End index of continuation for each sequence -/
  endIndices : Array Nat
  deriving Repr

/-- Batch sequences for multiple choice (common prefix) -/
def batchSequencesMC (tokenize : String → Array UInt64) (prompts : Array String) (bosToken : UInt64)
    : BatchInfo := Id.run do
  let mut tokens := #[]
  for prompt in prompts do
    tokens := tokens.push (#[bosToken] ++ tokenize prompt)

  let commonPrefix := findCommonPrefixLength tokens
  let startIndices := tokens.map (fun _ => commonPrefix)
  let endIndices := tokens.map (·.size)

  { tokens, startIndices, endIndices }

/-- Batch sequences for schema (common suffix) -/
def batchSequencesSchema (tokenize : String → Array UInt64) (prompts : Array String) (bosToken : UInt64)
    : BatchInfo := Id.run do
  let mut tokens := #[]
  for prompt in prompts do
    tokens := tokens.push (#[bosToken] ++ tokenize prompt)

  let commonSuffix := findCommonSuffixLength tokens
  let endIndices := tokens.map (·.size)
  let startIndices := endIndices.map (· - commonSuffix)

  { tokens, startIndices, endIndices }

/-- Batch sequences for language modeling -/
def batchSequencesLM (tokenize : String → Array UInt64) (promptWithout promptWith : String) (bosToken : UInt64)
    : BatchInfo :=
  let tokensWithout := #[bosToken] ++ tokenize promptWithout
  let tokensWith := #[bosToken] ++ tokenize promptWith

  { tokens := #[tokensWith]
    startIndices := #[tokensWithout.size]
    endIndices := #[tokensWith.size] }

/-! ## Evaluation -/

/-- Result of evaluating a single example -/
structure EvalResult where
  /-- Whether the prediction was correct -/
  correct : Bool
  /-- Loss values for each option (MC/schema) or continuation tokens (LM) -/
  losses : Array Float
  /-- Predicted index (MC/schema) or tokens (not used for LM) -/
  prediction : Nat
  deriving Repr

/-- Compute per-token losses from model output.
    Returns losses tensor where losses[i] is loss for predicting token i+1 from position i. -/
def computeLosses {batch seq vocab : UInt64} (logits : T #[batch, seq, vocab])
    (targetIds : T #[batch, seq]) : T #[batch, seq] :=
  -- Compute cross-entropy at each position
  -- logits: [batch, seq, vocab], targets: [batch, seq]
  -- Result: [batch, seq] (loss at each position)
  let batchSeq := batch * seq
  let logitsFlat := reshape logits #[batchSeq, vocab]
  let targetsFlat := reshape targetIds #[batchSeq]
  let lossesFlat := nn.cross_entropy_none logitsFlat targetsFlat
  reshape lossesFlat #[batch, seq]

/-- Evaluate multiple choice example.
    Returns index of choice with lowest average continuation loss. -/
def evaluateMC {batch seq vocab : UInt64}
    (logits : T #[batch, seq, vocab])
    (batchInfo : BatchInfo)
    (goldIdx : Nat) : EvalResult := Id.run do
  -- For each option, compute average loss over continuation tokens
  let mut meanLosses := #[]

  -- Note: In actual implementation, we would extract losses per option
  -- and compute means. For now, return placeholder.
  for _i in [:batchInfo.tokens.size] do
    -- Would compute: mean(losses[i, start:end])
    meanLosses := meanLosses.push (0.0 : Float)

  -- Predict option with lowest loss
  let mut minVal : Float := 1e30  -- Large value as infinity placeholder
  let mut minIdx := 0
  for i in [:meanLosses.size] do
    if meanLosses[i]! < minVal then
      minVal := meanLosses[i]!
      minIdx := i

  { correct := minIdx == goldIdx
    losses := meanLosses
    prediction := minIdx }

/-- Evaluate language modeling example.
    Returns whether all continuation tokens were predicted correctly. -/
def evaluateLM {seq vocab : UInt64}
    (logits : T #[1, seq, vocab])
    (batchInfo : BatchInfo) : EvalResult := Id.run do
  let startIdx := batchInfo.startIndices[0]!
  let endIdx := batchInfo.endIndices[0]!

  -- Get argmax predictions at each position
  -- predictions[i] predicts token at position i+1
  let predictions := nn.argmax logits 2  -- [1, seq]

  -- Check if predictions match actual tokens
  -- Would compare: predictions[0, start-1:end-1] vs tokens[0, start:end]
  -- For now, return placeholder
  { correct := false
    losses := #[]
    prediction := 0 }

/-! ## High-Level API -/

/-- Evaluation configuration -/
structure EvalConfig where
  /-- Model's maximum sequence length (for truncation) -/
  maxSeqLen : Option UInt64 := none
  /-- BOS token ID for padding -/
  bosToken : UInt64 := 0
  /-- Device to run on -/
  device : Device := Device.CPU
  deriving Repr, Inhabited

/-- Evaluate a single task across many examples.
    Returns accuracy (fraction correct). -/
def evaluateTask
    (examples : Array EvalItem)
    (taskMeta : TaskMeta)
    (tokenize : String → Array UInt64)
    (runModel : T #[] → IO (T #[]))  -- Placeholder for generic model forward
    (config : EvalConfig)
    : IO Float := do
  if examples.isEmpty then return 0.0

  -- Get distributed info
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed
    then dist.getRankAndWorldSize
    else pure (0, 1)

  let mut numCorrect : Nat := 0
  let mut numTotal : Nat := 0

  -- Evaluate examples strided by rank
  for idx in [:examples.size] do
    if idx % worldSize.toNat == rank.toNat then
      -- For now, placeholder evaluation
      -- Real implementation would:
      -- 1. Render prompts based on task type
      -- 2. Tokenize and batch
      -- 3. Run model forward
      -- 4. Compute losses and determine correctness
      numTotal := numTotal + 1

  -- Aggregate across ranks if distributed
  if isDistributed && worldSize > 1 then
    dist.barrier
    -- Would all-reduce numCorrect here

  return numCorrect.toFloat / numTotal.toFloat

/-- Predefined CORE tasks from DCLM benchmark -/
def coreTasks : Array TaskMeta := #[
  -- ARC-Easy and ARC-Challenge
  { taskType := .multipleChoice, numFewshot := 0, taskName := "arc_easy" },
  { taskType := .multipleChoice, numFewshot := 0, taskName := "arc_challenge" },
  -- HellaSwag
  { taskType := .multipleChoice, numFewshot := 0, taskName := "hellaswag" },
  -- Winogrande
  { taskType := .schema, numFewshot := 0, taskName := "winogrande" },
  -- PIQA
  { taskType := .multipleChoice, numFewshot := 0, taskName := "piqa" },
  -- SciQ
  { taskType := .multipleChoice, numFewshot := 0, taskName := "sciq" },
  -- BoolQ
  { taskType := .multipleChoice, numFewshot := 0, taskName := "boolq" },
  -- LAMBADA
  { taskType := .languageModeling, numFewshot := 0, taskName := "lambada" }
]

/-- Aggregate CORE score across tasks (average accuracy) -/
def computeCOREScore (taskAccuracies : Array (String × Float)) : Float :=
  if taskAccuracies.isEmpty then 0.0
  else
    let sum := taskAccuracies.foldl (fun acc (_, acc') => acc + acc') 0.0
    sum / taskAccuracies.size.toFloat

end torch.Eval.CORE
