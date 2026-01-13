/-
  Tyr/Generator/Engine.lean

  Main generation engine for autoregressive text generation with tool support.

  Features:
  - Token-by-token generation with KV caching
  - Tool invocation and result injection
  - Batch generation with per-row state tracking
  - Temperature and top-p/top-k sampling

  Based on nanochat's engine.py implementation.
-/
import Tyr.Torch
import Tyr.Generator.State
import Tyr.Generator.KVCache
import Tyr.Generator.Tools

namespace torch.Generator.Engine

open torch
open torch.data
open State
open KVCache
open Tools

/-- Sampling configuration -/
structure SamplingConfig where
  /-- Temperature for softmax (higher = more random) -/
  temperature : Float := 1.0
  /-- Top-p (nucleus) sampling threshold -/
  topP : Float := 0.9
  /-- Top-k sampling (0 = disabled) -/
  topK : Nat := 0
  /-- Maximum number of tokens to generate -/
  maxTokens : Nat := 256
  /-- Stop generation on these token IDs -/
  stopTokens : Array TokenId := #[]
  deriving Repr, Inhabited

/-- Generation engine wrapping model and tokenizer -/
structure Engine where
  /-- Special token IDs -/
  specialTokens : SpecialTokens
  /-- Tool registry -/
  toolRegistry : ToolRegistry
  /-- Sampling configuration -/
  samplingConfig : SamplingConfig
  deriving Inhabited

/-- Apply temperature scaling to logits -/
def applyTemperature {s : Shape} (logits : T s) (temp : Float) : T s :=
  if temp == 1.0 then logits
  else mul_scalar logits (1.0 / temp)

/-- Sample from logits using argmax (greedy).
    Returns array of token IDs, one per batch element. -/
def sampleGreedy {batch vocabSize : UInt64} (logits : T #[batch, vocabSize])
    : IO (Array TokenId) := do
  -- Get argmax along vocab dimension - returns [batch] tensor of indices
  let indices := nn.argmax logits 1
  -- Convert tensor to array of token IDs (use dynamic version since shape is computed)
  tensorToUInt64Array' (reshape indices #[])

/-- Apply top-k filtering to logits.
    Sets all logits outside the top k to -infinity. -/
def applyTopK {s : Shape} (logits : T s) (k : Nat) : T s :=
  if k == 0 then logits
  else nn.topKFilter logits k.toUInt64

/-- Apply top-p (nucleus) filtering to logits.
    Sets logits outside the nucleus (cumulative prob > p) to -infinity. -/
def applyTopP {s : Shape} (logits : T s) (p : Float) : T s :=
  if p >= 1.0 then logits
  else nn.topPFilter logits p

/-- Sample from logits with temperature, top-k, and top-p.
    Returns array of token IDs, one per batch element. -/
def sampleWithConfig {batch vocabSize : UInt64}
    (logits : T #[batch, vocabSize]) (cfg : SamplingConfig)
    : IO (Array TokenId) := do
  -- Apply temperature
  let scaled := applyTemperature logits cfg.temperature
  -- Apply top-k filtering
  let filtered := applyTopK scaled cfg.topK
  -- Apply top-p filtering
  let filtered := applyTopP filtered cfg.topP
  -- Convert to probabilities
  let probs := nn.softmax filtered (-1)
  -- Sample from distribution
  let samples ← nn.multinomial probs 1
  -- Squeeze to remove the sample dimension and convert to token IDs
  let squeezed := nn.squeezeDim samples (-1)
  tensorToUInt64Array' squeezed

/-- Process a single generated token for a row, handling tool invocation -/
def processToken (engine : Engine) (state : RowState) (token : TokenId)
    (decode : Array TokenId → String) (encode : String → Array TokenId)
    : IO RowState := do
  -- Check for end token
  if engine.specialTokens.isEndToken token then
    return state.markComplete.appendToken token

  -- Check for tool start
  match engine.specialTokens.isToolStart token with
  | some toolName =>
    return state.enterToolBlock toolName |>.appendToken token
  | none =>
    -- Check for tool end (if in tool block)
    if state.inToolBlock && engine.specialTokens.isToolEnd token then
      -- Execute the tool
      let toolInput := decode state.toolInputTokens
      let result ← engine.toolRegistry.dispatch toolInput

      -- Encode result with markers
      let resultStr := if result.success then result.output else s!"Error: {result.error.getD "unknown"}"
      let outputTokens := #[engine.specialTokens.outputStart] ++
                          encode resultStr ++
                          #[engine.specialTokens.outputEnd]

      return state.exitToolBlock outputTokens |>.appendToken token
    else if state.inToolBlock then
      -- Accumulate tool input
      return state.accumulateToolToken token |>.appendToken token
    else
      -- Normal token
      return state.appendToken token

/-- Generate tokens for a batch of prompts.

    This is the main generation loop that:
    1. Runs the model to get logits
    2. Samples next tokens using temperature/top-p/top-k
    3. Processes tokens for tool invocation
    4. Updates state and KV cache
    5. Repeats until max tokens or all rows complete

    Parameters:
    - engine: Generation engine with config and tools
    - promptTokens: Array of prompt token arrays, one per batch element
    - runModel: Model callback that takes input token IDs and returns logits tensor [batch, vocabSize]
    - decode: Decoder function for tool execution
    - encode: Encoder function for tool output injection -/
def generate (engine : Engine) (batch vocabSize : UInt64)
    (promptTokens : Array (Array TokenId))
    (runModel : Array TokenId → IO (T #[batch, vocabSize]))
    (decode : Array TokenId → String)
    (encode : String → Array TokenId)
    : IO (Array (Array TokenId)) := do
  let batchSize := promptTokens.size

  -- Initialize batch state
  let mut batchState := BatchState.fromPrompts promptTokens

  -- Main generation loop
  for _ in [:engine.samplingConfig.maxTokens] do
    -- Check if all rows are complete
    if batchState.allComplete then break

    -- Get last tokens for each row (for next prediction)
    let lastTokensArr := batchState.rows.map fun row =>
      row.lastToken.getD 0

    -- Run model to get logits [batch, vocabSize]
    let logits ← runModel lastTokensArr

    -- Sample next tokens using configured sampling strategy
    let sampledTokens ← sampleWithConfig logits engine.samplingConfig

    -- Get next tokens (either forced from tool output or sampled)
    let (nextTokens, newBatchState) := batchState.getNextTokens sampledTokens
    batchState := newBatchState

    -- Process each token for tool handling
    for i in [:batchSize] do
      let row := batchState.rows[i]!
      if !row.completed then
        let newRow ← processToken engine row nextTokens[i]! decode encode
        batchState := batchState.updateRow i newRow

  -- Return generated sequences
  return batchState.rows.map (·.tokens)

/-- Generate a single completion from a prompt -/
def generateOne (engine : Engine) (vocabSize : UInt64)
    (promptTokens : Array TokenId)
    (runModel : Array TokenId → IO (T #[1, vocabSize]))
    (decode : Array TokenId → String)
    (encode : String → Array TokenId)
    : IO (Array TokenId) := do
  let results ← generate engine 1 vocabSize #[promptTokens] runModel decode encode
  return results[0]!

/-- Configuration for the generation engine -/
structure EngineConfig where
  /-- Number of transformer layers -/
  numLayers : UInt64
  /-- Batch size -/
  batchSize : UInt64
  /-- Maximum sequence length -/
  maxSeqLen : UInt64
  /-- Number of KV heads -/
  numKvHeads : UInt64
  /-- Head dimension -/
  headDim : UInt64
  /-- Sampling configuration -/
  sampling : SamplingConfig := {}
  deriving Repr, Inhabited

/-- Create engine with configuration -/
def Engine.create (cfg : EngineConfig) (specialTokens : SpecialTokens) : Engine := {
  specialTokens := specialTokens
  toolRegistry := ToolRegistry.default
  samplingConfig := cfg.sampling
}

/-- Simple interface: generate text from string prompt -/
def generateText (engine : Engine) (vocabSize : UInt64)
    (prompt : String)
    (encode : String → Array TokenId)
    (decode : Array TokenId → String)
    (runModel : Array TokenId → IO (T #[1, vocabSize]))
    : IO String := do
  let promptTokens := encode prompt
  let outputTokens ← generateOne engine vocabSize promptTokens runModel decode encode
  -- Decode only the generated part (after prompt)
  let generatedTokens := outputTokens.extract promptTokens.size outputTokens.size
  return decode generatedTokens

end torch.Generator.Engine
