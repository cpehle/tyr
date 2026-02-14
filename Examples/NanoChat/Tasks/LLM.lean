/-
  Tyr/Tasks/LLM.lean

  LLM-specific task implementations for instruction tuning and evaluation.

  Based on nanochat's task system:
  - ARC (AI2 Reasoning Challenge) - multiple choice
  - MMLU (Massive Multitask Language Understanding) - multiple choice
  - GSM8K (Grade School Math) - generative with tool use
  - SmolTalk (general chat conversations)
  - CustomJSON (local JSONL conversations)
  - SpellingBee - generative with character counting
  - SimpleSpelling - generative spelling practice
  - HumanEval - code generation with execution-based evaluation

  Tasks support:
  - Loading from JSONL files
  - Evaluation (categorical and generative)
  - Reward computation for RL
-/
import Tyr.Data.Task
import Tyr.Data.TaskClass
import Lean.Data.Json

namespace torch.Tasks.LLM

open torch.Data.Task
open torch.Data.TaskClass (EvalTask EvalType EvalResult BoxedTask boxTask MixtureEntry entry)

/-! ## String Helpers -/

/-- Find position of substring in string, returns index into original string -/
def findSubstr (haystack : String) (needle : String) : Option Nat := Id.run do
  let hs := haystack.toList
  let ns := needle.toList
  if ns.isEmpty then return some 0
  for i in [:hs.length] do
    let mut found := true
    for j in [:ns.length] do
      if i + j >= hs.length then
        found := false
        break
      if hs[i + j]! != ns[j]! then
        found := false
        break
    if found then return some i
  return none

private def getObjArr? (json : Lean.Json) (field : String) : Option (Array Lean.Json) := do
  let value ← (json.getObjVal? field).toOption
  match value with
  | .arr xs => some xs
  | _ => none

private def parseRole? (role : String) : Option Role :=
  match role.trimAscii.toString.toLower with
  | "system" => some .system
  | "user" => some .user
  | "assistant" => some .assistant
  | "tool" => some .tool
  | _ => none

private def parseMessageJson? (json : Lean.Json) : Option Message := do
  let roleStr ← (json.getObjValAs? String "role").toOption
  let content ← (json.getObjValAs? String "content").toOption
  let role ← parseRole? roleStr
  return { role, content, parts := #[] }

private def validateAlternatingUserAssistant (msgs : Array Message) : Bool :=
  if msgs.isEmpty then
    false
  else
    let offset :=
      match msgs[0]? with
      | some first =>
        if first.role == .system then 1 else 0
      | none => 0
    let rest := msgs.extract offset msgs.size
    if rest.size < 2 then
      false
    else
      Id.run do
        let mut ok := true
        for i in [:rest.size] do
          let expected := if i % 2 == 0 then Role.user else Role.assistant
          match rest[i]? with
          | some msg =>
            if msg.role != expected then
              ok := false
          | none => ok := false
        ok

/-! ## Answer Extraction -/

/-- Extract numerical answer after "#### " marker (GSM8K/SpellingBee format) -/
def extractHashAnswer (response : String) : Option String := Id.run do
  -- Find "#### " marker
  let marker := "#### "
  let idx := findSubstr response marker
  match idx with
  | none => return none
  | some pos =>
    -- Extract everything after the marker until newline or end
    let start := pos + marker.length
    let rest := (response.drop start).toString
    -- Take until newline or end
    let answer := (rest.takeWhile (fun c => c != '\n' && c != '\r')).toString
    -- Normalize: remove commas, trim whitespace
    let normalized := answer.trimAscii.toString.replace "," ""
    if normalized.isEmpty then none else some normalized

/-- Extract single letter answer (A, B, C, D) for multiple choice -/
def extractLetterAnswer (response : String) (validLetters : Array String := #["A", "B", "C", "D"])
    : Option String := Id.run do
  let trimmed := response.trimAscii.toString
  -- Check if the response is just a letter
  if validLetters.contains trimmed then
    return some trimmed
  -- Otherwise look for first valid letter
  for letter in validLetters do
    if trimmed.startsWith letter then
      return some letter
  return none

/-! ## Multiple Choice Rendering -/

/-- Render a multiple choice question in nanochat format.
    Format: "- choice=letter" with letter AFTER choice for better token binding. -/
def renderMultipleChoice (question : String) (letters : Array String) (choices : Array String)
    : String := Id.run do
  let mut result := s!"Multiple Choice question: {question}\n"
  for (letter, choice) in letters.zip choices do
    result := result ++ s!"- {choice}={letter}\n"
  result := result ++ "\nRespond only with the letter of the correct answer."
  result

/-! ## Task Base Structure -/

/-- Base task configuration -/
structure TaskConfig where
  /-- Starting index for slicing -/
  start : Nat := 0
  /-- Ending index (none = all) -/
  stop : Option Nat := none
  /-- Step size for slicing -/
  step : Nat := 1
  /-- Random seed for shuffling -/
  seed : UInt64 := 42
  deriving Repr, Inhabited

/-! ## ARC Task (AI2 Reasoning Challenge) -/

/-- ARC example from dataset -/
structure ARCExample where
  question : String
  choices : Array String
  labels : Array String  -- e.g., ["A", "B", "C", "D"]
  answerKey : String     -- e.g., "A"
  deriving Repr, Inhabited

/-- Parse ARC example from JSON -/
def ARCExample.fromJson? (json : Lean.Json) : Option ARCExample := do
  let question ← (json.getObjValAs? String "question").toOption
  let choicesObj ← (json.getObjVal? "choices").toOption
  let choices ← (choicesObj.getObjValAs? (Array String) "text").toOption
  let labels ← (choicesObj.getObjValAs? (Array String) "label").toOption
  let answerKey ← (json.getObjValAs? String "answerKey").toOption
  return { question, choices, labels, answerKey }

/-- ARC task implementation -/
structure ARCTask where
  /-- Subset: "ARC-Easy" or "ARC-Challenge" -/
  subset : String
  /-- Split: "train", "validation", or "test" -/
  split : String
  /-- Loaded examples -/
  examples : Array ARCExample
  /-- Configuration -/
  config : TaskConfig
  deriving Repr, Inhabited

def ARCTask.evalType : EvalType := .categorical

def ARCTask.size (task : ARCTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def ARCTask.getExample (task : ARCTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  -- Render multiple choice question
  let userContent := renderMultipleChoice ex.question ex.labels ex.choices
  let messages := #[
    Message.user userContent,
    Message.assistant ex.answerKey
  ]
  return { messages, metadata := [("letters", String.intercalate "," ex.labels.toList)] }

def ARCTask.evaluate (_task : ARCTask) (conv : Conversation) (response : String) : EvalResult :=
  let expected := conv.messages.back?.map (·.content) |>.getD ""
  let predicted := extractLetterAnswer response |>.getD response.trimAscii.toString
  let correct := predicted == expected
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## MMLU Task -/

/-- MMLU example from dataset -/
structure MMLUExample where
  question : String
  choices : Array String
  answer : Nat          -- Index 0-3
  subject : String
  deriving Repr

def MMLUExample.fromJson? (json : Lean.Json) : Option MMLUExample := do
  -- HF `auxiliary_train` rows can be wrapped as `{ "train": {...} }`.
  let root := ((json.getObjVal? "train").toOption).getD json
  let question ← (root.getObjValAs? String "question").toOption
  let choices ← (root.getObjValAs? (Array String) "choices").toOption
  let answer :=
    match (root.getObjValAs? Nat "answer").toOption with
    | some a => some a
    | none =>
      match (root.getObjValAs? Int "answer").toOption with
      | some a => Int.toNat? a
      | none => none
  let answer ← answer
  let subject := (root.getObjValAs? String "subject").toOption |>.getD ""
  return { question, choices, answer, subject }

/-- MMLU task implementation -/
structure MMLUTask where
  subset : String
  split : String
  examples : Array MMLUExample
  config : TaskConfig
  deriving Repr

def MMLUTask.letters : Array String := #["A", "B", "C", "D"]

def MMLUTask.evalType : EvalType := .categorical

def MMLUTask.size (task : MMLUTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def MMLUTask.getExample (task : MMLUTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  let userContent := renderMultipleChoice ex.question MMLUTask.letters ex.choices
  let answerLetter := MMLUTask.letters[ex.answer]!
  let messages := #[
    Message.user userContent,
    Message.assistant answerLetter
  ]
  return {
    messages
    metadata := [("subject", ex.subject), ("letters", "A,B,C,D")]
  }

def MMLUTask.evaluate (_task : MMLUTask) (conv : Conversation) (response : String) : EvalResult :=
  let expected := conv.messages.back?.map (·.content) |>.getD ""
  let predicted := extractLetterAnswer response MMLUTask.letters |>.getD response.trimAscii.toString
  let correct := predicted == expected
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## SmolTalk Task -/

structure SmolTalkExample where
  messages : Array Message
  source : Option String := none
  deriving Repr

def SmolTalkExample.fromJson? (json : Lean.Json) : Option SmolTalkExample := do
  let rawMsgs ← getObjArr? json "messages"
  let msgs := rawMsgs.filterMap parseMessageJson?
  if msgs.size != rawMsgs.size then
    none
  else if !validateAlternatingUserAssistant msgs then
    none
  else
    let source := (json.getObjValAs? String "source").toOption
    some { messages := msgs, source := source }

structure SmolTalkTask where
  split : String
  examples : Array SmolTalkExample
  config : TaskConfig
  deriving Repr

def SmolTalkTask.evalType : EvalType := .generative

def SmolTalkTask.size (task : SmolTalkTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def SmolTalkTask.getExample (task : SmolTalkTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  let mut metadata : List (String × String) := []
  match ex.source with
  | some src => metadata := ("source", src) :: metadata
  | none => pure ()
  return { messages := ex.messages, metadata := metadata }

def SmolTalkTask.evaluate (_task : SmolTalkTask) (conv : Conversation) (response : String)
    : EvalResult :=
  let expected := conv.messages.back?.map (·.content) |>.getD ""
  let predicted := response.trimAscii.toString
  let correct := !predicted.isEmpty && !expected.isEmpty
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## CustomJSON Task -/

structure CustomJSONTask where
  filepath : String
  examples : Array Conversation
  config : TaskConfig
  deriving Repr

def CustomJSONTask.evalType : EvalType := .generative

def CustomJSONTask.size (task : CustomJSONTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def CustomJSONTask.getExample (task : CustomJSONTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  task.examples[physicalIdx]?

def CustomJSONTask.evaluate (_task : CustomJSONTask) (conv : Conversation) (response : String)
    : EvalResult :=
  let expected := conv.messages.back?.map (·.content) |>.getD ""
  let predicted := response.trimAscii.toString
  let correct := !predicted.isEmpty && !expected.isEmpty
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## GSM8K Task (Grade School Math) -/

/-- GSM8K example -/
structure GSM8KExample where
  question : String
  answer : String       -- Full solution with #### marker
  deriving Repr

def GSM8KExample.fromJson? (json : Lean.Json) : Option GSM8KExample := do
  let question ← (json.getObjValAs? String "question").toOption
  let answer ← (json.getObjValAs? String "answer").toOption
  return { question, answer }

/-- Parse GSM8K answer into parts (text and tool calls) -/
def parseGSM8KAnswer (answer : String) : Array ContentPart := Id.run do
  let mut parts : Array ContentPart := #[]
  let mut remaining := answer

  -- Simple parsing: split on << and >>
  while true do
    match findSubstr remaining "<<" with
    | none =>
      -- No more tool calls, add remaining text
      if !remaining.isEmpty then
        parts := parts.push { type := .text, content := remaining }
      break
    | some startIdx =>
      -- Add text before tool call
      let textBefore := (remaining.take startIdx).toString
      if !textBefore.isEmpty then
        parts := parts.push { type := .text, content := textBefore }

      -- Find end of tool call
      let afterStart := (remaining.drop (startIdx + 2)).toString
      match findSubstr afterStart ">>" with
      | none =>
        -- Malformed, add rest as text
        parts := parts.push { type := .text, content := (remaining.drop startIdx).toString }
        break
      | some endIdx =>
        -- Parse tool call: expr=result
        let inner := (afterStart.take endIdx).toString
        let (expr, result) := match inner.splitOn "=" with
          | [e, r] => (e, r)
          | [e] => (e, "")
          | _ => (inner, "")
        parts := parts.push { type := .toolCall "calculator", content := expr }
        parts := parts.push { type := .toolResult, content := result }
        remaining := (afterStart.drop (endIdx + 2)).toString

  parts

/-- GSM8K task implementation -/
structure GSM8KTask where
  subset : String
  split : String
  examples : Array GSM8KExample
  config : TaskConfig
  deriving Repr

def GSM8KTask.evalType : EvalType := .generative

def GSM8KTask.size (task : GSM8KTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def GSM8KTask.getExample (task : GSM8KTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  -- Parse answer into parts for tool use
  let answerParts := parseGSM8KAnswer ex.answer
  let messages := #[
    Message.user ex.question,
    { role := .assistant, content := ex.answer, parts := answerParts }
  ]
  return { messages }

def GSM8KTask.evaluate (_task : GSM8KTask) (conv : Conversation) (response : String) : EvalResult :=
  let expected := conv.messages.back?.bind (fun m => extractHashAnswer m.content) |>.getD ""
  let predicted := extractHashAnswer response |>.getD ""
  let correct := predicted == expected && !expected.isEmpty
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

def GSM8KTask.reward (task : GSM8KTask) (conv : Conversation) (response : String) : Float :=
  let result := task.evaluate conv response
  result.score

/-! ## SpellingBee Task -/

/-- Simple LCG random number generator -/
def lcgRandom (seed : UInt64) : UInt64 :=
  seed * 6364136223846793005 + 1442695040888963407

/-- Get random element from array -/
def randomChoice {α : Type} (arr : Array α) (seed : UInt64) : Option α :=
  if arr.isEmpty then none
  else arr[seed.toNat % arr.size]?

/-- Letters of the alphabet -/
def alphabet : Array Char :=
  #['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

/-- User message templates for SpellingBee -/
def spellingBeeTemplates : Array String := #[
  "How many {letter} are in the word {word}",
  "How many {letter} are in {word}",
  "Count the number of {letter} in {word}",
  "How many times does {letter} appear in {word}",
  "In the word {word}, how many {letter} are there",
  "Count how many {letter} appear in {word}",
  "How many {letter}s are in {word}",
  "Number of {letter} in {word}"
]

/-- SpellingBee task - count letter occurrences -/
structure SpellingBeeTask where
  /-- Word list -/
  words : Array String
  /-- Number of examples to generate -/
  size : Nat
  /-- Split: "train" or "test" -/
  split : String
  /-- Configuration -/
  config : TaskConfig
  deriving Repr

def SpellingBeeTask.evalType : EvalType := .generative

def SpellingBeeTask.getExample (task : SpellingBeeTask) (index : Nat) : Option Conversation := do
  let seed := if task.split == "test"
    then 10000000 + index.toUInt64
    else index.toUInt64

  -- Pick random word
  let wordSeed := lcgRandom seed
  let word ← randomChoice task.words wordSeed

  -- Pick letter (90% from word, 10% random)
  let letterSeed := lcgRandom wordSeed
  let useWordLetter := (letterSeed % 10) < 9
  let letterCharSeed := lcgRandom letterSeed
  let letter ← if useWordLetter then
    randomChoice word.toList.toArray letterCharSeed
  else
    randomChoice alphabet letterCharSeed

  -- Count occurrences
  let count := word.toList.filter (· == letter) |>.length

  -- Generate user message
  let templateSeed := lcgRandom letterCharSeed
  let template ← randomChoice spellingBeeTemplates templateSeed
  let userMsg := template.replace "{letter}" letter.toString |>.replace "{word}" word

  -- Generate assistant response with manual counting
  let wordLetters := String.intercalate "," (word.toList.map (·.toString))
  let mut manualText := s!"Let me count the '{letter}' in '{word}'.\n\n"
  manualText := manualText ++ s!"Spelling: {word}:{wordLetters}\n\n"

  let mut runningCount := 0
  let chars := word.toList
  for i in [:chars.length] do
    let c := chars[i]!
    if c == letter then
      runningCount := runningCount + 1
      manualText := manualText ++ s!"{i+1}:{c} hit! count={runningCount}\n"
    else
      manualText := manualText ++ s!"{i+1}:{c}\n"

  manualText := manualText ++ s!"\n#### {count}"

  let messages := #[
    Message.user userMsg,
    Message.assistant manualText
  ]

  return { messages }

def SpellingBeeTask.evaluate (_task : SpellingBeeTask) (conv : Conversation) (response : String)
    : EvalResult :=
  let expected := conv.messages.back?.bind (fun m => extractHashAnswer m.content) |>.getD ""
  let predicted := extractHashAnswer response |>.getD ""
  let correct := predicted == expected && !expected.isEmpty
  { correct, score := if correct then 1.0 else 0.0, expected, predicted }

/-! ## SimpleSpelling Task -/

/-- SimpleSpelling task - just spell words -/
structure SimpleSpellingTask where
  words : Array String
  size : Nat
  split : String
  config : TaskConfig
  deriving Repr

def SimpleSpellingTask.evalType : EvalType := .generative

def SimpleSpellingTask.getExample (task : SimpleSpellingTask) (index : Nat) : Option Conversation := do
  let seed := if task.split == "test"
    then 10000000 + index.toUInt64
    else index.toUInt64

  let wordSeed := lcgRandom seed
  let word ← randomChoice task.words wordSeed
  let wordLetters := String.intercalate "," (word.toList.map (·.toString))

  let messages := #[
    Message.user s!"Spell the word: {word}",
    Message.assistant s!"{word}:{wordLetters}"
  ]

  return { messages }

/-! ## Task Loading from JSONL -/

/-- Load ARC examples from JSONL file -/
def loadARCFromJsonl (path : System.FilePath) : IO (Array ARCExample) := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut examples : Array ARCExample := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json =>
      if let some ex := ARCExample.fromJson? json then
        examples := examples.push ex
    | .error _ => pure ()
  return examples

/-- Load MMLU examples from JSONL file -/
def loadMMLUFromJsonl (path : System.FilePath) : IO (Array MMLUExample) := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut examples : Array MMLUExample := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json =>
      if let some ex := MMLUExample.fromJson? json then
        examples := examples.push ex
    | .error _ => pure ()
  return examples

/-- Load GSM8K examples from JSONL file -/
def loadGSM8KFromJsonl (path : System.FilePath) : IO (Array GSM8KExample) := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut examples : Array GSM8KExample := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json =>
      if let some ex := GSM8KExample.fromJson? json then
        examples := examples.push ex
    | .error _ => pure ()
  return examples

/-- Load SmolTalk examples from JSONL file. -/
def loadSmolTalkFromJsonl (path : System.FilePath) : IO (Array SmolTalkExample) := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut examples : Array SmolTalkExample := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json =>
      if let some ex := SmolTalkExample.fromJson? json then
        examples := examples.push ex
    | .error _ => pure ()
  return examples

/-- Load conversations from CustomJSON JSONL format (one JSON array of messages per line). -/
def loadCustomJSONConversations (path : System.FilePath) : IO (Array Conversation) := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut convs : Array Conversation := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok (.arr rawMsgs) =>
      let msgs := rawMsgs.filterMap parseMessageJson?
      if msgs.size == rawMsgs.size && validateAlternatingUserAssistant msgs then
        convs := convs.push { messages := msgs }
    | .ok _ => pure ()
    | .error _ => pure ()
  return convs

/-- Load word list from text file (one word per line) -/
def loadWordList (path : System.FilePath) : IO (Array String) := do
  let content ← IO.FS.readFile path
  let words := content.splitOn "\n"
    |>.map (fun s => s.trimAscii.toString)
    |>.filter (!·.isEmpty)
  return words.toArray

/-! ## Task Factory Functions -/

/-- Create ARC task from JSONL file -/
def createARCTask (path : System.FilePath) (subset : String) (split : String)
    (config : TaskConfig := {}) : IO ARCTask := do
  let examples ← loadARCFromJsonl path
  return { subset, split, examples, config }

/-- Create MMLU task from JSONL file -/
def createMMLUTask (path : System.FilePath) (subset : String) (split : String)
    (config : TaskConfig := {}) : IO MMLUTask := do
  let examples ← loadMMLUFromJsonl path
  return { subset, split, examples, config }

/-- Create GSM8K task from JSONL file -/
def createGSM8KTask (path : System.FilePath) (subset : String) (split : String)
    (config : TaskConfig := {}) : IO GSM8KTask := do
  let examples ← loadGSM8KFromJsonl path
  return { subset, split, examples, config }

/-- Create SmolTalk task from JSONL file. -/
def createSmolTalkTask (path : System.FilePath) (split : String)
    (config : TaskConfig := {}) : IO SmolTalkTask := do
  let examples ← loadSmolTalkFromJsonl path
  return { split, examples, config }

/-- Create CustomJSON task from a local JSONL file. -/
def createCustomJSONTask (path : System.FilePath)
    (config : TaskConfig := {}) : IO CustomJSONTask := do
  let examples ← loadCustomJSONConversations path
  return { filepath := path.toString, examples, config }

/-- Create SpellingBee task from word list -/
def createSpellingBeeTask (wordListPath : System.FilePath) (size : Nat) (split : String)
    (config : TaskConfig := {}) : IO SpellingBeeTask := do
  let words ← loadWordList wordListPath
  return { words, size, split, config }

/-- Create SimpleSpelling task from word list -/
def createSimpleSpellingTask (wordListPath : System.FilePath) (size : Nat) (split : String)
    (config : TaskConfig := {}) : IO SimpleSpellingTask := do
  let words ← loadWordList wordListPath
  return { words, size, split, config }

/-! ## HumanEval Task (Code Generation) -/

/-- HumanEval example from dataset -/
structure HumanEvalExample where
  /-- Task ID (e.g., "HumanEval/0") -/
  taskId : String
  /-- Function signature and docstring prompt -/
  prompt : String
  /-- Function entry point name -/
  entryPoint : String
  /-- Canonical solution -/
  canonical : String
  /-- Test code to verify correctness -/
  test : String
  deriving Repr, Inhabited

def HumanEvalExample.fromJson? (json : Lean.Json) : Option HumanEvalExample := do
  let taskId ← (json.getObjValAs? String "task_id").toOption
  let prompt ← (json.getObjValAs? String "prompt").toOption
  let entryPoint ← (json.getObjValAs? String "entry_point").toOption
  let canonical ← (json.getObjValAs? String "canonical_solution").toOption
  let test ← (json.getObjValAs? String "test").toOption
  return { taskId, prompt, entryPoint, canonical, test }

/-- HumanEval task implementation -/
structure HumanEvalTask where
  /-- Loaded examples -/
  examples : Array HumanEvalExample
  /-- Configuration -/
  config : TaskConfig
  deriving Repr, Inhabited

def HumanEvalTask.evalType : EvalType := .generative

def HumanEvalTask.size (task : HumanEvalTask) : Nat :=
  let stop := task.config.stop.getD task.examples.size
  let span := stop - task.config.start
  (span + task.config.step - 1) / task.config.step

def HumanEvalTask.getExample (task : HumanEvalTask) (index : Nat) : Option Conversation := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  -- User provides the prompt, assistant generates the solution
  let messages := #[
    Message.user s!"Complete the following Python function:\n\n{ex.prompt}",
    Message.assistant ex.canonical
  ]
  return {
    messages
    metadata := [
      ("task_id", ex.taskId),
      ("entry_point", ex.entryPoint)
    ]
  }

/-- Get the prompt for generation (without solution) -/
def HumanEvalTask.getPrompt (task : HumanEvalTask) (index : Nat) : Option String := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  return ex.prompt

/-- Get test code for an example -/
def HumanEvalTask.getTest (task : HumanEvalTask) (index : Nat) : Option (String × String) := do
  let physicalIdx := task.config.start + index * task.config.step
  let ex ← task.examples[physicalIdx]?
  return (ex.entryPoint, ex.test)

/-- Evaluate generated code (checks if code executes correctly).
    Note: Actual execution requires the Execution module.
    This just checks format. -/
def HumanEvalTask.evaluate (_task : HumanEvalTask) (_conv : Conversation) (response : String)
    : EvalResult :=
  -- For HumanEval, we can't really evaluate without execution
  -- Just check that response contains code (use findSubstr from this file)
  let hasDef := (findSubstr response "def ").isSome
  let hasReturn := (findSubstr response "return").isSome
  let hasCode := hasDef || hasReturn
  let truncated := response.take 100
  {
    correct := hasCode
    score := if hasCode then 0.5 else 0.0  -- Partial credit for having code
    expected := ""
    predicted := truncated.toString
  }

/-- Load HumanEval examples from JSONL file -/
def loadHumanEvalFromJsonl (path : System.FilePath) : IO (Array HumanEvalExample) := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n" |>.filter (!·.isEmpty)
  let mut examples : Array HumanEvalExample := #[]
  for line in lines do
    match Lean.Json.parse line with
    | .ok json =>
      if let some ex := HumanEvalExample.fromJson? json then
        examples := examples.push ex
    | .error _ => pure ()
  return examples

/-- Create HumanEval task from JSONL file -/
def createHumanEvalTask (path : System.FilePath) (config : TaskConfig := {})
    : IO HumanEvalTask := do
  let examples ← loadHumanEvalFromJsonl path
  return { examples, config }

/-! ## Task Typeclass Instances -/

/-- ARC task implements Task typeclass -/
instance : EvalTask ARCTask where
  name := "arc"
  evalType _ := .categorical
  size := ARCTask.size
  getExample := ARCTask.getExample
  evaluate := ARCTask.evaluate

/-- MMLU task implements Task typeclass -/
instance : EvalTask MMLUTask where
  name := "mmlu"
  evalType _ := .categorical
  size := MMLUTask.size
  getExample := MMLUTask.getExample
  evaluate := MMLUTask.evaluate

/-- GSM8K task implements Task typeclass -/
instance : EvalTask GSM8KTask where
  name := "gsm8k"
  evalType _ := .generative
  size := GSM8KTask.size
  getExample := GSM8KTask.getExample
  evaluate := GSM8KTask.evaluate
  reward := GSM8KTask.reward

/-- SmolTalk task implements Task typeclass. -/
instance : EvalTask SmolTalkTask where
  name := "smoltalk"
  evalType _ := .generative
  size := SmolTalkTask.size
  getExample := SmolTalkTask.getExample
  evaluate := SmolTalkTask.evaluate

/-- CustomJSON task implements Task typeclass. -/
instance : EvalTask CustomJSONTask where
  name := "customjson"
  evalType _ := .generative
  size := CustomJSONTask.size
  getExample := CustomJSONTask.getExample
  evaluate := CustomJSONTask.evaluate

/-- SpellingBee task implements Task typeclass -/
instance : EvalTask SpellingBeeTask where
  name := "spelling_bee"
  evalType _ := .generative
  size t := t.size
  getExample := SpellingBeeTask.getExample
  evaluate := SpellingBeeTask.evaluate

/-- SimpleSpelling task implements Task typeclass -/
instance : EvalTask SimpleSpellingTask where
  name := "simple_spelling"
  evalType _ := .generative
  size t := t.size
  getExample := SimpleSpellingTask.getExample
  evaluate _ _ response := { correct := true, score := 1.0, expected := "", predicted := response }

/-- HumanEval task implements Task typeclass -/
instance : EvalTask HumanEvalTask where
  name := "humaneval"
  evalType _ := .generative
  size := HumanEvalTask.size
  getExample := HumanEvalTask.getExample
  evaluate := HumanEvalTask.evaluate

/-! ## Unified Task Interface (Legacy)

The AnyTask union is kept for backward compatibility.
New code should prefer using the Task typeclass via BoxedTask.
-/

/-- Generic task that wraps concrete implementations -/
inductive AnyTask where
  | arc : ARCTask → AnyTask
  | mmlu : MMLUTask → AnyTask
  | gsm8k : GSM8KTask → AnyTask
  | smolTalk : SmolTalkTask → AnyTask
  | customJSON : CustomJSONTask → AnyTask
  | spellingBee : SpellingBeeTask → AnyTask
  | simpleSpelling : SimpleSpellingTask → AnyTask
  | humanEval : HumanEvalTask → AnyTask
  deriving Repr, Inhabited

def AnyTask.evalType : AnyTask → EvalType
  | .arc _ => ARCTask.evalType
  | .mmlu _ => MMLUTask.evalType
  | .gsm8k _ => GSM8KTask.evalType
  | .smolTalk _ => SmolTalkTask.evalType
  | .customJSON _ => CustomJSONTask.evalType
  | .spellingBee _ => SpellingBeeTask.evalType
  | .simpleSpelling _ => SimpleSpellingTask.evalType
  | .humanEval _ => HumanEvalTask.evalType

def AnyTask.size : AnyTask → Nat
  | .arc t => t.size
  | .mmlu t => t.size
  | .gsm8k t => t.size
  | .smolTalk t => t.size
  | .customJSON t => t.size
  | .spellingBee t => t.size
  | .simpleSpelling t => t.size
  | .humanEval t => t.size

def AnyTask.getExample (task : AnyTask) (index : Nat) : Option Conversation :=
  match task with
  | .arc t => t.getExample index
  | .mmlu t => t.getExample index
  | .gsm8k t => t.getExample index
  | .smolTalk t => t.getExample index
  | .customJSON t => t.getExample index
  | .spellingBee t => t.getExample index
  | .simpleSpelling t => t.getExample index
  | .humanEval t => t.getExample index

def AnyTask.evaluate (task : AnyTask) (conv : Conversation) (response : String) : EvalResult :=
  match task with
  | .arc t => t.evaluate conv response
  | .mmlu t => t.evaluate conv response
  | .gsm8k t => t.evaluate conv response
  | .smolTalk t => t.evaluate conv response
  | .customJSON t => t.evaluate conv response
  | .spellingBee t => t.evaluate conv response
  | .simpleSpelling _ =>
    -- SimpleSpelling doesn't have formal evaluation
    { correct := true, score := 1.0, expected := "", predicted := response }
  | .humanEval t => t.evaluate conv response

/-- Compute reward for RL (0.0 or 1.0 for most tasks) -/
def AnyTask.reward (task : AnyTask) (conv : Conversation) (response : String) : Float :=
  let result := task.evaluate conv response
  result.score

/-! ## Task Mixture for Training -/

/-- Entry in a task mixture with sampling weight -/
structure AnyMixtureEntry where
  task : AnyTask
  weight : Nat := 1
  deriving Repr, Inhabited

/-- Create unified task mixture from multiple tasks -/
structure AnyTaskMixture where
  entries : Array AnyMixtureEntry
  /-- Shuffled index map: global idx → (task idx, local idx) -/
  indexMap : Array (Nat × Nat)
  seed : UInt64
  deriving Repr

def AnyTaskMixture.create (entries : Array AnyMixtureEntry) (seed : UInt64 := 42)
    : AnyTaskMixture := Id.run do
  -- Build index map
  let mut indices : Array (Nat × Nat) := #[]
  for taskIdx in [:entries.size] do
    let entry := entries[taskIdx]!
    let taskSize := entry.task.size
    for _ in [:entry.weight] do
      for localIdx in [:taskSize] do
        indices := indices.push (taskIdx, localIdx)

  -- Deterministic shuffle
  let mut shuffled := indices
  let mut state := seed
  for i in [:(indices.size - 1)] do
    state := lcgRandom state
    let range := indices.size - i
    let j := i + (state % range.toUInt64).toNat
    let tmp := shuffled[i]!
    shuffled := shuffled.set! i shuffled[j]!
    shuffled := shuffled.set! j tmp

  { entries, indexMap := shuffled, seed }

def AnyTaskMixture.size (mix : AnyTaskMixture) : Nat := mix.indexMap.size

def AnyTaskMixture.getExample (mix : AnyTaskMixture) (index : Nat) : Option Conversation := do
  let (taskIdx, localIdx) ← mix.indexMap[index]?
  let entry ← mix.entries[taskIdx]?
  entry.task.getExample localIdx

def AnyTaskMixture.evaluate (mix : AnyTaskMixture) (index : Nat) (response : String)
    : Option EvalResult := do
  let (taskIdx, localIdx) ← mix.indexMap[index]?
  let entry ← mix.entries[taskIdx]?
  let conv ← entry.task.getExample localIdx
  return entry.task.evaluate conv response

end torch.Tasks.LLM
