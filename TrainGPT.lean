/-
  GPT Training Script

  Train a tiny GPT model on Shakespeare character data.
  Uses the same data format as nanoGPT.
-/
import Tyr
import Tyr.Train

open torch
open torch.gpt
open torch.train

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

/-- Train with data of known size -/
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

def main : IO Unit := do
  IO.println "Starting..."
  IO.println "=== Tyr GPT Training ==="
  IO.println ""

  -- Configuration - scaled down nanoGPT for CPU
  -- Same architecture (6 layers, 6 heads) but smaller dimensions
  let modelCfg := Config.medium_shakespeare
  IO.println s!"Model config: vocab={modelCfg.vocab_size}, block={modelCfg.block_size}, embd={modelCfg.n_embd}, heads={modelCfg.n_head}, layers={modelCfg.n_layer}"

  let trainCfg : TrainConfig := {
    maxIters := 1000
    logInterval := 100
    learningRate := 1e-3
    warmupIters := 100
    batchSize := 8
    blockSize := modelCfg.block_size
  }

  -- Try to load Shakespeare data, fall back to random data
  IO.println ""
  IO.println "Loading data..."

  -- Try to load real data first
  let fileExists ← do
    try
      let _ ← data.binFileTokenCount "data/shakespeare_char/train.bin"
      pure true
    catch _ =>
      pure false

  let trainedParams ← if fileExists then
    IO.println "File exists, getting token count..."
    let n ← data.binFileTokenCount "data/shakespeare_char/train.bin"
    IO.println s!"Found {n} tokens in Shakespeare data"
    IO.println "Loading tokens..."
    let tokens ← data.loadU16Bin n "data/shakespeare_char/train.bin"
    IO.println "Loaded Shakespeare training data!"
    runTraining modelCfg trainCfg tokens
  else
    IO.println "Shakespeare data not found, using random tokens for testing..."
    let numTokens : UInt64 := 10000
    let tokens ← randint 0 modelCfg.vocab_size.toInt64 #[numTokens]
    runTraining modelCfg trainCfg tokens

  -- Generate some text
  IO.println ""
  IO.println "=== Generating Shakespeare ==="
  IO.println ""

  let prompt := "ROMEO:"
  let promptTokens := encode prompt
  IO.println s!"Prompt: \"{prompt}\""

  let generated ← generate trainedParams promptTokens 200 1.0
  let text := decode generated

  IO.println "Generated text:"
  IO.println "---"
  IO.println text
  IO.println "---"
  IO.println ""
  IO.println "Done!"
