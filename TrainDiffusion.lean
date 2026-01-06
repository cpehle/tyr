/-
  Diffusion Training Script

  Train a tiny discrete diffusion model on Shakespeare text.
  Uses masked diffusion with bidirectional attention.
-/
import Tyr
import Tyr.Diffusion
import Tyr.DiffusionSchedule
import Tyr.DiffusionTrain

open torch
open torch.diffusion
open torch.diffusion.train
open torch.nanoproof (RotaryCache)

-- ASCII encoding/decoding (simple, like tiny-diffusion)
-- Vocab is 0-127 where 0 is reserved as [MASK] token
def asciiEncode (s : String) : Array Int64 :=
  s.data.toArray.map fun c =>
    let code := c.toNat
    if code < 128 then code.toInt64 else 127  -- clamp to ASCII

def asciiDecode (tokens : Array Int64) : String :=
  String.mk <| tokens.toList.map fun t =>
    let code := t.toUInt64.toNat
    if code == 0 then '[' -- Show mask token as [
    else if code < 128 then Char.ofNat code
    else '?'

/-- Convert tensor to array for decoding -/
def tensorToArray {n : UInt64} (t : T #[n]) : IO (Array Int64) := do
  let mut arr := #[]
  for i in [:n.toNat] do
    -- Use slice1d to get single element (index_select_1d has issues)
    let elem : T #[1] := data.slice1d t i.toInt64 (i.toInt64 + 1)
    let val := nn.itemInt elem
    arr := arr.push val
  return arr

/-- Train diffusion model with validation -/
def runTrainingWithVal {nTrain nVal : UInt64}
    (modelCfg : Config) (trainCfg : TrainConfig)
    (trainData : T #[nTrain]) (valData : T #[nVal])
    : IO (DiffusionParams modelCfg) := do
  IO.println ""
  IO.println "Initializing diffusion model..."
  let params ← DiffusionParams.init modelCfg

  -- Initialize optimizer state
  let opt := Optim.adamw (lr := trainCfg.learningRate) (weight_decay := trainCfg.weightDecay)
  let optState := opt.init params

  -- Initialize mask schedule
  let schedule := MaskedDiffusionSchedule.init modelCfg.diffusion_steps modelCfg.mask_token_id modelCfg.context_len

  -- Initialize rotary cache (exact seq_len to avoid slice2d issues)
  let rotaryCache ← RotaryCache.init modelCfg.seq_len modelCfg.headDim

  IO.println s!"Model initialized with {modelCfg.n_layer} layers, {modelCfg.n_embd} embedding dim"
  IO.println s!"Diffusion steps: {modelCfg.diffusion_steps}, context_len: {modelCfg.context_len}"
  IO.println ""

  -- Train with validation
  let (finalParams, _bestValLoss) ← trainLoop trainCfg params optState schedule trainData valData rotaryCache

  IO.println ""
  return finalParams

/-- Train on random data for testing -/
def runTrainingRandom (modelCfg : Config) (trainCfg : TrainConfig) : IO (DiffusionParams modelCfg) := do
  IO.println ""
  IO.println "Using random tokens for testing..."
  let numTokens : UInt64 := 50000
  -- Generate random tokens (1-127, avoiding 0 which is mask)
  let trainData ← randint 1 127 #[numTokens]
  let valData ← randint 1 127 #[numTokens / 10]

  runTrainingWithVal modelCfg trainCfg trainData valData

/-- Load text file and convert to ASCII tokens -/
def loadTextFile (path : String) : IO (Array Int64) := do
  let contents ← IO.FS.readFile path
  return asciiEncode contents

def main : IO Unit := do
  IO.println "Starting..."
  IO.println "=== Tyr Discrete Diffusion Training ==="
  IO.println ""
  (← IO.getStdout).flush

  -- Small model config (matches nanoGPT style)
  let modelCfg : Config := Config.small
  IO.println s!"Model config: vocab={modelCfg.vocab_size}, seq_len={modelCfg.seq_len}"
  IO.println s!"  n_layer={modelCfg.n_layer}, n_head={modelCfg.n_head}, n_embd={modelCfg.n_embd}"
  IO.println s!"  diffusion_steps={modelCfg.diffusion_steps}, context_len={modelCfg.context_len}"

  let trainCfg : TrainConfig := {
    maxIters := 2000
    evalInterval := 500
    logInterval := 100
    learningRate := 1e-2
    minLr := 1e-4
    warmupIters := 200
    lrDecayIters := 2000
    gradClip := 1.0
    batchSize := 64
    weightDecay := 0.0
  }

  -- Try to load Shakespeare data
  IO.println ""
  IO.println "Loading data..."
  (← IO.getStdout).flush

  let shakespeareExists ← do
    try
      let _ ← IO.FS.readFile "data/shakespeare_char/input.txt"
      pure true
    catch _ =>
      pure false

  let trainedParams ← if shakespeareExists then
    IO.println "Found Shakespeare text file..."
    let tokens ← loadTextFile "data/shakespeare_char/input.txt"
    IO.println s!"Loaded {tokens.size} tokens"

    -- Split 90/10 for train/val
    let splitIdx := tokens.size * 9 / 10
    let trainTokens := tokens.extract 0 splitIdx
    let valTokens := tokens.extract splitIdx tokens.size

    let nTrain : UInt64 := trainTokens.size.toUInt64
    let nVal : UInt64 := valTokens.size.toUInt64

    IO.println s!"Training set: {nTrain} tokens"
    IO.println s!"Validation set: {nVal} tokens"

    -- Convert to tensors
    let trainData : T #[nTrain] := reshape (data.fromInt64Array trainTokens) #[nTrain]
    let valData : T #[nVal] := reshape (data.fromInt64Array valTokens) #[nVal]

    runTrainingWithVal modelCfg trainCfg trainData valData
  else
    IO.println "Shakespeare data not found, using random tokens for testing..."
    runTrainingRandom modelCfg trainCfg

  -- Generate some text using the trained model
  IO.println ""
  IO.println "=== Generating Text ==="
  IO.println ""

  -- Initialize rotary cache for generation (exact seq_len)
  let rotaryCache ← RotaryCache.init modelCfg.seq_len modelCfg.headDim

  -- Generate a batch of 1 using top-k decoding (decode k=4 tokens per step)
  IO.println "Generating with top-k decoding (k=4)..."
  let generated ← sampleTopK (batch := 1) trainedParams rotaryCache 4 0.8 256

  -- Extract first sequence and decode
  let genFlat := reshape generated #[modelCfg.seq_len]
  let genArr ← tensorToArray genFlat
  let text := asciiDecode genArr

  IO.println "Generated text:"
  IO.println "---"
  IO.println text
  IO.println "---"
  IO.println ""
  IO.println "Done!"
