/-
  Examples/NanoChat/Eval/COREData.lean

  Load CORE evaluation data from the eval bundle.

  The CORE evaluation bundle contains JSONL files for each task:
  - arc_easy.jsonl, arc_challenge.jsonl
  - hellaswag.jsonl
  - winogrande.jsonl
  - piqa.jsonl
  - sciq.jsonl
  - boolq.jsonl
  - lambada.jsonl

  Each file contains evaluation examples in a task-specific format.
-/
import Tyr.Data.Download
import Tyr.Data.HuggingFace
import Examples.NanoChat.Eval.CORE
import Lean.Data.Json

namespace torch.Eval.COREData

open torch.Data.Download
open torch.Data.HuggingFace
open torch.Eval.CORE

/-! ## CORE Task Configuration -/

/-- Configuration for a CORE evaluation task -/
structure CORETaskConfig where
  /-- Task label (e.g., "arc_easy") -/
  label : String
  /-- Task type (MC, schema, LM) -/
  taskType : TaskType
  /-- Path to dataset file within eval bundle -/
  datasetPath : String
  /-- Number of few-shot examples -/
  numFewshot : Nat := 0
  /-- Delimiter between context and continuation -/
  continuationDelimiter : String := " "
  deriving Repr, Inhabited

/-- Default CORE task configurations matching the DCLM benchmark -/
def defaultCOREConfigs : Array CORETaskConfig := #[
  -- ARC (AI2 Reasoning Challenge)
  { label := "arc_easy"
    taskType := .multipleChoice
    datasetPath := "arc_easy.jsonl"
    numFewshot := 0
    continuationDelimiter := "\nAnswer: " },
  { label := "arc_challenge"
    taskType := .multipleChoice
    datasetPath := "arc_challenge.jsonl"
    numFewshot := 0
    continuationDelimiter := "\nAnswer: " },
  -- HellaSwag
  { label := "hellaswag"
    taskType := .multipleChoice
    datasetPath := "hellaswag.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- Winogrande
  { label := "winogrande"
    taskType := .schema
    datasetPath := "winogrande.jsonl"
    numFewshot := 0
    continuationDelimiter := "" },
  -- PIQA
  { label := "piqa"
    taskType := .multipleChoice
    datasetPath := "piqa.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- SciQ
  { label := "sciq"
    taskType := .multipleChoice
    datasetPath := "sciq.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- BoolQ
  { label := "boolq"
    taskType := .multipleChoice
    datasetPath := "boolq.jsonl"
    numFewshot := 0
    continuationDelimiter := " " },
  -- LAMBADA
  { label := "lambada"
    taskType := .languageModeling
    datasetPath := "lambada.jsonl"
    numFewshot := 0
    continuationDelimiter := " " }
]

/-! ## JSON Parsing Utilities -/

/-- Parse ARC example from JSON -/
def parseARCItem (json : Lean.Json) : Option MCItem := do
  let question ← getJsonString json "question"
  let choicesObj ← (json.getObjVal? "choices").toOption
  let choices ← getJsonStringArray choicesObj "text"
  let labels ← getJsonStringArray choicesObj "label"
  let answerKey ← getJsonString json "answerKey"
  -- Find gold index from answerKey
  let goldIdx := labels.findIdx? (· == answerKey) |>.getD 0
  return { query := question, choices, gold := goldIdx }

/-- Parse HellaSwag example from JSON.
    Format: {"ctx": str, "endings": [str], "label": str} -/
def parseHellaSwagItem (json : Lean.Json) : Option MCItem := do
  let ctx ← getJsonString json "ctx"
  let endings ← getJsonStringArray json "endings"
  let labelStr ← getJsonString json "label"
  let gold := labelStr.toNat?.getD 0
  return { query := ctx, choices := endings, gold }

/-- Parse Winogrande example from JSON.
    Format: {"sentence": str, "option1": str, "option2": str, "answer": str} -/
def parseWinograndeItem (json : Lean.Json) : Option SchemaItem := do
  let sentence ← getJsonString json "sentence"
  let option1 ← getJsonString json "option1"
  let option2 ← getJsonString json "option2"
  let answer ← getJsonString json "answer"
  -- Replace _ with options to create context options
  let ctx1 := sentence.replace "_" option1
  let ctx2 := sentence.replace "_" option2
  -- answer is "1" or "2"
  let gold := if answer == "1" then 0 else 1
  -- The continuation is empty for Winogrande (schema task)
  return { contextOptions := #[ctx1, ctx2], continuation := "", gold }

/-- Parse PIQA example from JSON.
    Format: {"goal": str, "sol1": str, "sol2": str, "label": int} -/
def parsePIQAItem (json : Lean.Json) : Option MCItem := do
  let goal ← getJsonString json "goal"
  let sol1 ← getJsonString json "sol1"
  let sol2 ← getJsonString json "sol2"
  let gold ← getJsonNat json "label"
  return { query := goal, choices := #[sol1, sol2], gold }

/-- Parse SciQ example from JSON.
    Format: {"question": str, "distractor1": str, "distractor2": str, "distractor3": str,
             "correct_answer": str, "support": str} -/
def parseSciQItem (json : Lean.Json) : Option MCItem := do
  let question ← getJsonString json "question"
  let correct ← getJsonString json "correct_answer"
  let d1 ← getJsonString json "distractor1"
  let d2 ← getJsonString json "distractor2"
  let d3 ← getJsonString json "distractor3"
  -- Put correct answer first, then distractors
  return { query := question, choices := #[correct, d1, d2, d3], gold := 0 }

/-- Parse BoolQ example from JSON.
    Format: {"question": str, "passage": str, "answer": bool} -/
def parseBoolQItem (json : Lean.Json) : Option MCItem := do
  let question ← getJsonString json "question"
  let passage ← getJsonString json "passage"
  let answer ← getJsonBool json "answer"
  let query := s!"{passage}\nQuestion: {question}"
  let gold := if answer then 0 else 1  -- Yes=0, No=1
  return { query, choices := #["Yes", "No"], gold }

/-- Parse LAMBADA example from JSON.
    Format: {"text": str} - last word is the target -/
def parseLAMBADAItem (json : Lean.Json) : Option LMItem := do
  let text ← getJsonString json "text"
  -- Split off last word as continuation
  let words := text.splitOn " "
  if words.length < 2 then none
  let context := String.intercalate " " (words.dropLast)
  let continuation := words.getLast!
  return { context, continuation }

/-! ## Loading CORE Data -/

/-- Load CORE task config from metadata file.
    If no metadata file exists, uses default configs. -/
def loadCOREConfig (bundleDir : String) : IO (Array CORETaskConfig) := do
  -- Check if a config file exists
  let configPath := s!"{bundleDir}/core_config.json"
  if ← fileExists configPath then
    -- Parse config file
    let content ← IO.FS.readFile ⟨configPath⟩
    match Lean.Json.parse content with
    | .error _ => return defaultCOREConfigs
    | .ok json =>
      -- Parse array of configs (simplified - in practice would parse full structure)
      return defaultCOREConfigs
  else
    return defaultCOREConfigs

/-- Load CORE task data from eval bundle.
    Returns array of EvalItem for the specified task. -/
def loadCORETaskData (bundleDir : String) (taskConfig : CORETaskConfig)
    : IO (Array EvalItem) := do
  let dataPath := s!"{bundleDir}/{taskConfig.datasetPath}"

  -- Check if file exists
  if !(← fileExists dataPath) then
    IO.eprintln s!"Warning: CORE data file not found: {dataPath}"
    return #[]

  -- Read JSONL file
  let jsons ← loadJsonlFromFile dataPath

  -- Parse based on task type
  let mut items : Array EvalItem := #[]
  for json in jsons do
    match taskConfig.taskType with
    | .multipleChoice =>
      let parsed := match taskConfig.label with
        | "arc_easy" | "arc_challenge" => parseARCItem json
        | "hellaswag" => parseHellaSwagItem json
        | "piqa" => parsePIQAItem json
        | "sciq" => parseSciQItem json
        | "boolq" => parseBoolQItem json
        | _ => none
      if let some item := parsed then
        items := items.push (.mc item)
    | .schema =>
      if let some item := parseWinograndeItem json then
        items := items.push (.schema item)
    | .languageModeling =>
      if let some item := parseLAMBADAItem json then
        items := items.push (.lm item)

  IO.println s!"  Loaded {items.size} examples for {taskConfig.label}"
  return items

/-- Find CORE config for a given TaskMeta -/
def coreTaskMetaToConfig (taskMeta : TaskMeta) (configs : Array CORETaskConfig)
    : Option CORETaskConfig :=
  configs.find? fun cfg =>
    cfg.label == taskMeta.taskName || cfg.label.replace "_" "" == taskMeta.taskName.replace "_" ""

/-! ## Loading All CORE Tasks -/

/-- Load all CORE tasks and return a map from task name to examples -/
def loadAllCORETasks (bundleDir : String)
    : IO (Array (String × Array EvalItem)) := do
  let configs ← loadCOREConfig bundleDir
  let mut results : Array (String × Array EvalItem) := #[]

  for config in configs do
    let items ← loadCORETaskData bundleDir config
    results := results.push (config.label, items)

  return results

/-- Load CORE data for evaluation.
    Downloads the eval bundle if necessary and returns the bundle directory. -/
def prepareCOREEvalData (cacheDir : String := "~/.cache/nanochat") : IO String := do
  -- Ensure eval bundle is downloaded
  let bundleDir ← ensureCOREData cacheDir
  return bundleDir

/-! ## Rendering Helpers

These functions convert EvalItem to prompts for model evaluation.
-/

/-- Render prompts for a multiple choice item.
    Returns one prompt per choice option. -/
def renderPromptsMCFromItem (item : EvalItem) (taskMeta : TaskMeta) : Array String :=
  match item with
  | .mc mcItem =>
    let delimiter := taskMeta.continuationDelimiter
    mcItem.choices.map fun choice =>
      mcItem.query ++ delimiter ++ choice
  | _ => #[]

/-- Get gold index for multiple choice item -/
def getGoldIdxMC (item : EvalItem) : Nat :=
  match item with
  | .mc mcItem => mcItem.gold
  | _ => 0

/-- Render prompts for a schema item.
    Returns one prompt per context option. -/
def renderPromptsSchemaFromItem (item : EvalItem) (taskMeta : TaskMeta) : Array String :=
  match item with
  | .schema schemaItem =>
    let delimiter := taskMeta.continuationDelimiter
    schemaItem.contextOptions.map fun ctx =>
      ctx ++ delimiter ++ schemaItem.continuation
  | _ => #[]

/-- Get gold index for schema item -/
def getGoldIdxSchema (item : EvalItem) : Nat :=
  match item with
  | .schema schemaItem => schemaItem.gold
  | _ => 0

/-- Render prompts for language modeling item.
    Returns (context, context + continuation). -/
def renderPromptsLMFromItem (item : EvalItem) (_meta : TaskMeta) : (String × String) :=
  match item with
  | .lm lmItem =>
    (lmItem.context, lmItem.context ++ " " ++ lmItem.continuation)
  | _ => ("", "")

/-! ## Few-Shot Support -/

/-- Get few-shot examples for a task.
    Returns array of example items for few-shot prompting. -/
def getFewshotExamples (items : Array EvalItem) (numFewshot : Nat) (seed : UInt64 := 42)
    : Array EvalItem := Id.run do
  if numFewshot == 0 || items.isEmpty then return #[]

  -- Simple deterministic selection: first N items
  let n := min numFewshot items.size
  return items.extract 0 n

/-- Render few-shot prefix for multiple choice tasks -/
def renderFewshotPrefixMC (examples : Array EvalItem) (delimiter : String) : String := Id.run do
  let mut result := ""
  for item in examples do
    match item with
    | .mc mcItem =>
      let answer := mcItem.choices[mcItem.gold]!
      result := result ++ mcItem.query ++ delimiter ++ answer ++ "\n\n"
    | _ => pure ()
  result

end torch.Eval.COREData
