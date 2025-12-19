import Tyr

open torch
open torch.gpt
open torch.train
open torch.checkpoint

-- Shakespeare character vocabulary (65 chars)
-- This must match the prepare.py encoding
def shakespeareChars : String :=
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def charToInt (c : Char) : Int64 :=
  match shakespeareChars.data.findIdx? (· == c) with
  | some idx => idx.toInt64
  | none => 0  -- fallback to newline

def intToChar (i : Int64) : Char :=
  shakespeareChars.data.getD i.toUInt64.toNat '\n'

def encode (s : String) : Array Int64 :=
  s.data.toArray.map charToInt

def decode (tokens : Array Int64) : String :=
  String.mk (tokens.toList.map intToChar)

/-- Train with data of known size (legacy, no validation) -/
def runTraining {n : UInt64} (modelCfg : Config) (trainCfg : TrainConfig)
    (trainData : T #[n]) : IO (GPTParams modelCfg) := do
  IO.println ""
  IO.println "Initializing model..."
  let params ← GPTParams.init modelCfg
  let optState := GPTOptState.init modelCfg

  IO.println s!"Model initialized with {modelCfg.n_layer} layers"
  IO.println ""

  -- Train
  let finalParams ← trainLoop trainCfg params optState trainData

  IO.println ""
  return finalParams

/-- Train with validation data and checkpointing -/
def runTrainingWithVal {nTrain nVal : UInt64} (modelCfg : Config) (trainCfg : TrainConfig)
    (trainData : T #[nTrain]) (valData : T #[nVal])
    (checkpointDir : Option String := none) : IO (GPTParams modelCfg) := do
  IO.println ""
  IO.println "Initializing model..."
  let params ← GPTParams.init modelCfg
  let optState := GPTOptState.init modelCfg

  IO.println s!"Model initialized with {modelCfg.n_layer} layers"
  IO.println ""

  -- Train with validation
  let (finalParams, bestValLoss) ← trainLoopWithVal trainCfg params optState trainData valData

  -- Save final checkpoint if directory specified
  if let some dir := checkpointDir then
    saveCheckpoint finalParams trainCfg.maxIters bestValLoss 0.0 dir

  IO.println ""
  return finalParams

def main : IO Unit := do
  IO.println "Starting..."
  IO.println "=== Tyr GPT Training ==="
  IO.println ""
  (← IO.getStdout).flush

  -- nanoGPT CPU configuration for Shakespeare character-level training
  let modelCfg := Config.nanogpt_cpu_shakespeare
  IO.println s!"Model config: vocab={modelCfg.vocab_size}, block={modelCfg.block_size}, embd={modelCfg.n_embd}, heads={modelCfg.n_head}, layers={modelCfg.n_layer}, dropout={modelCfg.dropout}"

  let trainCfg : TrainConfig := {
    maxIters := 2000
    evalInterval := 200
    logInterval := 100
    learningRate := 1e-3
    minLr := 1e-4
    warmupIters := 100
    lrDecayIters := 2000
    gradClip := 1.0
    batchSize := 12
    blockSize := modelCfg.block_size
  }

  IO.println ""
  IO.println "Loading data..."
  (← IO.getStdout).flush

  let trainFileExists ← do
    try
      let _ ← data.binFileTokenCount "data/shakespeare_char/train.bin"
      pure true
    catch _ =>
      pure false

  let valFileExists ← do
    try
      let _ ← data.binFileTokenCount "data/shakespeare_char/val.bin"
      pure true
    catch _ =>
      pure false

  let trainedParams ← if trainFileExists then
    IO.println "Found Shakespeare data..."
    let nTrain ← data.binFileTokenCount "data/shakespeare_char/train.bin"
    IO.println s!"Training set: {nTrain} tokens"
    let trainData ← data.loadU16Bin nTrain "data/shakespeare_char/train.bin"

    if valFileExists then
      let nVal ← data.binFileTokenCount "data/shakespeare_char/val.bin"
      IO.println s!"Validation set: {nVal} tokens"
      let valData ← data.loadU16Bin nVal "data/shakespeare_char/val.bin"
      IO.println "Training with validation..."
      runTrainingWithVal modelCfg trainCfg trainData valData (some "checkpoints/gpt")
    else
      IO.println "No validation data found, training without validation..."
      runTraining modelCfg trainCfg trainData
  else
    IO.println "Shakespeare data not found, using random tokens for testing..."
    let numTokens : UInt64 := 10000
    let tokens ← randint 0 modelCfg.vocab_size.toInt64 #[numTokens]
    runTraining modelCfg trainCfg tokens

  IO.println ""
  IO.println "Done!"