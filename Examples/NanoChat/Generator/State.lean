/-
  Tyr/Generator/State.lean

  Generation state machine for token-by-token generation with tool support.

  Based on nanochat's engine.py RowState implementation.
  Each row (batch element) maintains its own state for:
  - Current token sequence
  - Forced tokens (to inject tool outputs)
  - Tool invocation tracking
  - Completion status
-/
import Tyr.Torch

/-!
# `Examples.NanoChat.Generator.State`

NanoChat generation subsystem module for State.

## Overview
- Example module intended for runnable workflows and reference usage patterns.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace torch.Generator.State

open torch

/-- Token ID type (index into vocabulary) -/
abbrev TokenId := UInt64

/-- State for a single row (batch element) during generation -/
structure RowState where
  /-- Full token sequence generated so far -/
  tokens : Array TokenId
  /-- Queue of tokens to force-inject (tool outputs) -/
  forcedTokens : Array TokenId
  /-- Whether currently inside a tool invocation block -/
  inToolBlock : Bool
  /-- Tokens accumulated for tool input (between tool_start and tool_end) -/
  toolInputTokens : Array TokenId
  /-- Which tool is currently being invoked (if any) -/
  currentTool : Option String
  /-- Whether generation is complete for this row -/
  completed : Bool
  /-- Number of tokens generated (excluding prompt) -/
  generatedCount : Nat
  deriving Repr, Inhabited

/-- Create initial row state from prompt tokens -/
def RowState.fromPrompt (promptTokens : Array TokenId) : RowState := {
  tokens := promptTokens
  forcedTokens := #[]
  inToolBlock := false
  toolInputTokens := #[]
  currentTool := none
  completed := false
  generatedCount := 0
}

/-- Check if there are forced tokens to inject -/
def RowState.hasForcedTokens (state : RowState) : Bool :=
  state.forcedTokens.size > 0

/-- Pop the next forced token (if any) -/
def RowState.popForcedToken (state : RowState) : Option TokenId × RowState :=
  if state.forcedTokens.size > 0 then
    let token := state.forcedTokens[0]!
    let newForced := state.forcedTokens.extract 1 state.forcedTokens.size
    (some token, { state with forcedTokens := newForced })
  else
    (none, state)

/-- Append a token to the sequence -/
def RowState.appendToken (state : RowState) (token : TokenId) : RowState :=
  { state with
    tokens := state.tokens.push token
    generatedCount := state.generatedCount + 1
  }

/-- Enter tool invocation mode -/
def RowState.enterToolBlock (state : RowState) (toolName : String) : RowState :=
  { state with
    inToolBlock := true
    toolInputTokens := #[]
    currentTool := some toolName
  }

/-- Accumulate a token for tool input -/
def RowState.accumulateToolToken (state : RowState) (token : TokenId) : RowState :=
  { state with toolInputTokens := state.toolInputTokens.push token }

/-- Exit tool block and inject result tokens -/
def RowState.exitToolBlock (state : RowState) (resultTokens : Array TokenId) : RowState :=
  { state with
    inToolBlock := false
    toolInputTokens := #[]
    currentTool := none
    forcedTokens := state.forcedTokens ++ resultTokens
  }

/-- Mark generation as complete -/
def RowState.markComplete (state : RowState) : RowState :=
  { state with completed := true }

/-- Get the last token in the sequence -/
def RowState.lastToken (state : RowState) : Option TokenId :=
  state.tokens.back?

/-- Batch state for multiple rows -/
structure BatchState (batchSize : Nat) where
  /-- Per-row states -/
  rows : Array RowState
  /-- Total tokens generated across all rows -/
  totalGenerated : Nat
  deriving Repr

/-- Create batch state from array of prompts -/
def BatchState.fromPrompts (prompts : Array (Array TokenId)) : BatchState prompts.size := {
  rows := prompts.map RowState.fromPrompt
  totalGenerated := 0
}

/-- Check if all rows are complete -/
def BatchState.allComplete (state : BatchState n) : Bool :=
  state.rows.all (·.completed)

/-- Count completed rows -/
def BatchState.completedCount (state : BatchState n) : Nat :=
  state.rows.foldl (fun acc row => if row.completed then acc + 1 else acc) 0

/-- Get next token for each row (either forced or sampled) -/
def BatchState.getNextTokens (state : BatchState n) (sampledTokens : Array TokenId)
    : Array TokenId × BatchState n := Id.run do
  let mut nextTokens : Array TokenId := #[]
  let mut newRows : Array RowState := #[]
  for i in [:state.rows.size] do
    let row := state.rows[i]!
    let (forcedOpt, newRow) := row.popForcedToken
    match forcedOpt with
    | some forced =>
      nextTokens := nextTokens.push forced
      newRows := newRows.push newRow
    | none =>
      let sampled := sampledTokens[i]!
      nextTokens := nextTokens.push sampled
      newRows := newRows.push row
  return (nextTokens, { state with rows := newRows, totalGenerated := state.totalGenerated + n })

/-- Update row at index -/
def BatchState.updateRow (state : BatchState n) (idx : Nat) (newRow : RowState)
    : BatchState n :=
  if idx < state.rows.size then
    { state with rows := state.rows.set! idx newRow }
  else
    state

/-- Special token IDs (to be configured per tokenizer) -/
structure SpecialTokens where
  /-- End of text / sequence -/
  eot : TokenId
  /-- Beginning of text -/
  bot : TokenId
  /-- Tool invocation start -/
  toolStart : TokenId
  /-- Tool invocation end -/
  toolEnd : TokenId
  /-- Tool output start -/
  outputStart : TokenId
  /-- Tool output end -/
  outputEnd : TokenId
  /-- Calculator tool start -/
  calcStart : TokenId
  /-- Calculator tool end -/
  calcEnd : TokenId
  /-- Lean prover tool start -/
  proverStart : TokenId
  /-- Lean prover tool end -/
  proverEnd : TokenId
  deriving Repr, Inhabited

/-- Check if token is a tool start token, returns tool name if so -/
def SpecialTokens.isToolStart (tokens : SpecialTokens) (token : TokenId) : Option String :=
  if token == tokens.toolStart then some "generic"
  else if token == tokens.calcStart then some "calculator"
  else if token == tokens.proverStart then some "prover"
  else none

/-- Check if token is a tool end token -/
def SpecialTokens.isToolEnd (tokens : SpecialTokens) (token : TokenId) : Bool :=
  token == tokens.toolEnd || token == tokens.calcEnd || token == tokens.proverEnd

/-- Check if token signals end of generation -/
def SpecialTokens.isEndToken (tokens : SpecialTokens) (token : TokenId) : Bool :=
  token == tokens.eot

end torch.Generator.State
