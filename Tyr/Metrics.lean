/-
  Tyr/Metrics.lean

  Evaluation metrics for neural network training.
  Provides accuracy, precision, recall, F1 score, and other common metrics.

  Includes BPB (bits-per-byte) evaluation from nanochat's loss_eval.py:
  - Vocab-size-independent metric
  - Uses TokenBytes mapping for byte counts
  - Lower is better (English text approaches ~0.8)
-/
import Tyr.Torch
import Tyr.Tokenizer.TokenBytes

namespace torch.metrics

open torch

/-! ## Classification Metrics -/

/-- Compute accuracy: fraction of correct predictions.
    - preds: logits or probabilities [batch, num_classes]
    - targets: ground truth class indices [batch] -/
def accuracy {batch num_classes : UInt64}
    (preds : T #[batch, num_classes])
    (targets : T #[batch])
    : Float :=
  let predicted := nn.argmax preds 1  -- [batch]
  let isCorrect := eq predicted targets
  let matchesFloat := toFloat' isCorrect
  let numCorrect := nn.sumAll matchesFloat
  nn.item numCorrect / batch.toFloat

/-- Top-K accuracy: fraction where true class is in top K predictions.
    Note: This is an approximation that counts total matches across all top-k
    positions. For exact top-k accuracy with per-sample any(), use a custom
    implementation with reduction along the k dimension. -/
def topkAccuracy {batch num_classes : UInt64}
    (preds : T #[batch, num_classes])
    (targets : T #[batch])
    (k : UInt64)
    : Float :=
  -- Get top-k predictions
  let (_, topkIndices) := topk preds k 1  -- [batch, k]
  -- For each sample, check if target is in top-k
  -- We expand targets to [batch, k] and compare
  let targetsExpanded := nn.unsqueeze targets 1  -- [batch, 1]
  let targetsExpanded := nn.expand targetsExpanded #[batch, k]  -- [batch, k]
  -- Check equality
  let isCorrect := eq topkIndices targetsExpanded  -- [batch, k]
  -- Convert to float and count total matches
  let matchesFloat := toFloat' isCorrect  -- [batch, k]
  let totalMatches := nn.item (nn.sumAll matchesFloat)
  -- Clamp to batch size (each sample can have at most 1 correct target)
  let clampedMatches := if totalMatches > batch.toFloat then batch.toFloat else totalMatches
  clampedMatches / batch.toFloat

/-! ## Binary Classification Metrics -/

/-- Compute precision for binary classification.
    precision = TP / (TP + FP) -/
def precision {batch : UInt64}
    (preds : T #[batch])
    (targets : T #[batch])
    (threshold : Float := 0.5)
    : Float :=
  let thresholdT := full #[batch] threshold
  let binaryPreds := ge preds thresholdT
  let predictedPositive := toFloat' binaryPreds
  let actualPositive := toFloat' targets
  let truePositives := predictedPositive * actualPositive
  let tp := nn.item (nn.sumAll truePositives)
  let totalPredicted := nn.item (nn.sumAll predictedPositive)
  if totalPredicted < 1e-8 then 0.0
  else tp / totalPredicted

/-- Compute recall for binary classification.
    recall = TP / (TP + FN) -/
def recall {batch : UInt64}
    (preds : T #[batch])
    (targets : T #[batch])
    (threshold : Float := 0.5)
    : Float :=
  let thresholdT := full #[batch] threshold
  let binaryPreds := ge preds thresholdT
  let predictedPositive := toFloat' binaryPreds
  let actualPositive := toFloat' targets
  let truePositives := predictedPositive * actualPositive
  let tp := nn.item (nn.sumAll truePositives)
  let totalActual := nn.item (nn.sumAll actualPositive)
  if totalActual < 1e-8 then 0.0
  else tp / totalActual

/-- Compute F1 score: harmonic mean of precision and recall.
    F1 = 2 * (precision * recall) / (precision + recall) -/
def f1Score {batch : UInt64}
    (preds : T #[batch])
    (targets : T #[batch])
    (threshold : Float := 0.5)
    : Float :=
  let prec := precision preds targets threshold
  let recallVal := recall preds targets threshold
  let sum := prec + recallVal
  if sum < 1e-8 then 0.0
  else 2.0 * (prec * recallVal) / sum

/-- Compute F-beta score with configurable beta.
    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall) -/
def fbetaScore {batch : UInt64}
    (preds : T #[batch])
    (targets : T #[batch])
    (beta : Float := 1.0)
    (threshold : Float := 0.5)
    : Float :=
  let prec := precision preds targets threshold
  let recallVal := recall preds targets threshold
  let beta2 := beta * beta
  let denom := beta2 * prec + recallVal
  if denom < 1e-8 then 0.0
  else (1.0 + beta2) * (prec * recallVal) / denom

/-! ## Language Model Metrics -/

/-- Compute perplexity from cross-entropy loss.
    perplexity = exp(loss) -/
def perplexity (loss : Float) : Float := Float.exp loss

/-- Compute bits per character (BPC) from loss.
    BPC = loss / ln(2) -/
def bitsPerChar (loss : Float) : Float := loss / Float.log 2.0

/-- Compute bits per byte (BPB) from total nats and byte count.
    BPB = total_nats / (ln(2) * total_bytes)

    This is a vocab-size-independent metric for comparing
    language models trained with different tokenizers.
    Lower is better (approaching ~0.8 for English text).

    Based on nanochat's loss_eval.py:
    - total_nats: sum of cross-entropy losses (not averaged)
    - total_bytes: sum of UTF-8 bytes for valid tokens -/
def bitsPerByte (totalNats : Float) (totalBytes : UInt64) : Float :=
  if totalBytes == 0 then 0.0
  else totalNats / (Float.log 2.0 * totalBytes.toFloat)

/-! ## BPB Evaluation Accumulators -/

/-- Accumulator for bits-per-byte evaluation.
    Tracks total nats and total bytes across batches. -/
structure BPBAccumulator where
  /-- Sum of cross-entropy losses (in nats, not bits) -/
  totalNats : Float := 0.0
  /-- Sum of UTF-8 bytes for all valid tokens -/
  totalBytes : UInt64 := 0
  /-- Number of tokens processed -/
  numTokens : UInt64 := 0
  deriving Repr, Inhabited

/-- Add a batch result to the BPB accumulator -/
def BPBAccumulator.addBatch (acc : BPBAccumulator)
    (batchNats : Float) (batchBytes : UInt64) (batchTokens : UInt64)
    : BPBAccumulator :=
  { totalNats := acc.totalNats + batchNats
  , totalBytes := acc.totalBytes + batchBytes
  , numTokens := acc.numTokens + batchTokens
  }

/-- Compute the current BPB from the accumulator -/
def BPBAccumulator.bpb (acc : BPBAccumulator) : Float :=
  bitsPerByte acc.totalNats acc.totalBytes

/-- Compute average loss (nats per token) from the accumulator -/
def BPBAccumulator.avgLoss (acc : BPBAccumulator) : Float :=
  if acc.numTokens == 0 then 0.0
  else acc.totalNats / acc.numTokens.toFloat

/-- Reset the accumulator -/
def BPBAccumulator.reset : BPBAccumulator :=
  { totalNats := 0.0, totalBytes := 0, numTokens := 0 }

/-! ## Running Metrics (for aggregation over batches) -/

/-- Accumulator for running mean -/
structure RunningMean where
  sum : Float := 0.0
  count : Nat := 0
  deriving Repr, Inhabited

/-- Add a value to running mean -/
def RunningMean.add (rm : RunningMean) (value : Float) : RunningMean :=
  { sum := rm.sum + value, count := rm.count + 1 }

/-- Add multiple values to running mean -/
def RunningMean.addBatch (rm : RunningMean) (value : Float) (n : Nat) : RunningMean :=
  { sum := rm.sum + value * n.toFloat, count := rm.count + n }

/-- Get the current mean -/
def RunningMean.mean (rm : RunningMean) : Float :=
  if rm.count == 0 then 0.0 else rm.sum / rm.count.toFloat

/-- Reset the running mean -/
def RunningMean.reset : RunningMean := { sum := 0.0, count := 0 }

/-- Accumulator for confusion matrix (binary classification) -/
structure BinaryConfusionMatrix where
  tp : Nat := 0  -- True positives
  fp : Nat := 0  -- False positives
  tn : Nat := 0  -- True negatives
  fn : Nat := 0  -- False negatives
  deriving Repr, Inhabited

/-- Compute precision from confusion matrix -/
def BinaryConfusionMatrix.precision (cm : BinaryConfusionMatrix) : Float :=
  let total := cm.tp + cm.fp
  if total == 0 then 0.0 else cm.tp.toFloat / total.toFloat

/-- Compute recall from confusion matrix -/
def BinaryConfusionMatrix.recall (cm : BinaryConfusionMatrix) : Float :=
  let total := cm.tp + cm.fn
  if total == 0 then 0.0 else cm.tp.toFloat / total.toFloat

/-- Compute F1 from confusion matrix -/
def BinaryConfusionMatrix.f1 (cm : BinaryConfusionMatrix) : Float :=
  let prec := cm.precision
  let recallVal := cm.recall
  let sum := prec + recallVal
  if sum < 1e-8 then 0.0 else 2.0 * (prec * recallVal) / sum

/-- Compute accuracy from confusion matrix -/
def BinaryConfusionMatrix.accuracy (cm : BinaryConfusionMatrix) : Float :=
  let total := cm.tp + cm.fp + cm.tn + cm.fn
  if total == 0 then 0.0 else (cm.tp + cm.tn).toFloat / total.toFloat

/-! ## Distance Metrics -/

/-- Compute cosine similarity between two tensors -/
def cosineSimilarity {s : Shape}
    (a b : T s)
    : Float :=
  let dot := nn.sumAll (a * b)
  let normA := nn.sqrt (nn.sumAll (a * a))
  let normB := nn.sqrt (nn.sumAll (b * b))
  let denom := normA * normB
  let denomSafe := add_scalar denom 1e-8
  nn.item (nn.div dot denomSafe)

/-- Compute mean squared error -/
def mse {s : Shape} (pred target : T s) : Float :=
  let diff := pred - target
  let squared := diff * diff
  nn.item (nn.meanAll squared)

/-- Compute root mean squared error -/
def rmse {s : Shape} (pred target : T s) : Float :=
  Float.sqrt (mse pred target)

/-- Compute mean absolute error -/
def mae {s : Shape} (pred target : T s) : Float :=
  let diff := nn.abs (pred - target)
  nn.item (nn.meanAll diff)

/-! ## Full BPB Evaluation -/

/-- Configuration for BPB evaluation -/
structure BPBEvalConfig where
  /-- Number of evaluation steps -/
  numSteps : Nat := 100
  /-- Batch size (total across devices) -/
  batchSize : UInt64 := 32
  /-- Sequence length -/
  seqLen : UInt64 := 512
  /-- Ignore index for loss computation (typically pad token) -/
  ignoreIndex : Int64 := -1
  deriving Repr, Inhabited

/-- Result of BPB evaluation -/
structure BPBResult where
  /-- Bits per byte (main metric) -/
  bpb : Float
  /-- Average cross-entropy loss (in nats) -/
  avgLoss : Float
  /-- Total tokens processed -/
  totalTokens : UInt64
  /-- Total bytes processed -/
  totalBytes : UInt64
  /-- Perplexity (exp(avgLoss)) -/
  perplexity : Float
  deriving Repr

/-- Compute BPB from a single batch.

    Following nanochat's loss_eval.py:
    - loss2d: per-token loss from model (reduction='none')
    - targets: target token IDs
    - tokenBytes: TokenBytes mapping
    - ignoreIndex: tokens to ignore (typically -1 for padding)

    Returns (sum of nats, total bytes, valid token count) -/
def computeBPBBatch (b s : UInt64)
    (loss2d : T #[b, s])
    (targets : T #[b, s])
    (tokenBytes : tokenizer.TokenBytes)
    (_ignoreIndex : Int64 := -1)
    : IO (Float × UInt64 × UInt64) := do
  -- Create mask: valid = (targets >= 0) when ignoreIndex is -1
  -- or (targets != ignoreIndex) in general
  -- We use ge (>=) with threshold 0 for the common case of ignoreIndex = -1
  let zeroTensor := zeros #[b, s]
  let validMask := ge targets zeroTensor  -- [B, S] bool: targets >= 0

  -- Convert to float mask (1.0 for valid, 0.0 for invalid)
  let floatMask := toFloat' validMask  -- [B, S]

  -- Look up bytes per token using tokenBytes
  let totalBytes ← tokenBytes.totalBytesWithMask targets floatMask

  -- Only sum loss where mask > 0 AND bytes > 0
  -- Get bytes per token for masking
  let numTokens := b * s
  let flatIds := reshape targets #[numTokens]
  let flatBytes := reshape tokenBytes.bytes #[tokenBytes.vocabSize.toUInt64]
  let bytesPerToken := data.indexSelect flatBytes 0 flatIds
  let bytesPerToken2d := reshape bytesPerToken #[b, s]
  let bytesFloat := toFloat' bytesPerToken2d

  -- Create combined mask: valid AND has_bytes > 0
  let hasBytes := gt bytesFloat zeroTensor
  let hasBytesFloat := toFloat' hasBytes
  let combinedMask := floatMask * hasBytesFloat

  -- Sum loss weighted by combined mask
  let maskedLoss := loss2d * combinedMask
  let totalNats := nn.item (nn.sumAll maskedLoss)

  -- Count valid tokens
  let validTokens := nn.item (nn.sumAll combinedMask)

  return (totalNats, totalBytes, validTokens.toUInt64)

/-- Run full BPB evaluation over a validation loader.

    Parameters:
    - forwardFn: Model forward pass that returns per-token loss [B, S]
    - getBatchFn: Function to get next batch (inputs, targets)
    - tokenBytes: TokenBytes mapping
    - config: Evaluation configuration

    Returns BPB result with all metrics. -/
def evaluateBPB (b s : UInt64)
    (forwardFn : T #[b, s] → T #[b, s] → IO (T #[b, s]))  -- (input, target) → loss2d
    (getBatchFn : IO (T #[b, s] × T #[b, s]))  -- () → (inputs, targets)
    (tokenBytes : tokenizer.TokenBytes)
    (config : BPBEvalConfig)
    : IO BPBResult := do

  let mut acc := BPBAccumulator.reset

  for _ in [:config.numSteps] do
    let (inputs, targets) ← getBatchFn

    -- Forward pass to get per-token loss
    let loss2d ← forwardFn inputs targets

    -- Compute batch BPB contribution
    let (batchNats, batchBytes, batchTokens) ← computeBPBBatch b s
      loss2d targets tokenBytes config.ignoreIndex

    acc := acc.addBatch batchNats batchBytes batchTokens

  let avgLoss := acc.avgLoss
  return {
    bpb := acc.bpb
    avgLoss := avgLoss
    totalTokens := acc.numTokens
    totalBytes := acc.totalBytes
    perplexity := perplexity avgLoss
  }

/-- Compute BPB for a pre-loaded batch array (no loader needed).
    Useful for testing or small validation sets. -/
def evaluateBPBOnBatches (b s : UInt64)
    (forwardFn : T #[b, s] → T #[b, s] → IO (T #[b, s]))
    (batches : Array (T #[b, s] × T #[b, s]))
    (tokenBytes : tokenizer.TokenBytes)
    (ignoreIndex : Int64 := -1)
    : IO BPBResult := do

  let mut acc := BPBAccumulator.reset

  for (inputs, targets) in batches do
    let loss2d ← forwardFn inputs targets
    let (batchNats, batchBytes, batchTokens) ← computeBPBBatch b s
      loss2d targets tokenBytes ignoreIndex
    acc := acc.addBatch batchNats batchBytes batchTokens

  let avgLoss := acc.avgLoss
  return {
    bpb := acc.bpb
    avgLoss := avgLoss
    totalTokens := acc.numTokens
    totalBytes := acc.totalBytes
    perplexity := perplexity avgLoss
  }

/-- Log BPB evaluation results -/
def logBPBResult (result : BPBResult) (label : String := "Eval") : IO Unit := do
  let msg := s!"{label}: BPB={result.bpb} loss={result.avgLoss} ppl={result.perplexity} " ++
    s!"tokens={result.totalTokens} bytes={result.totalBytes}"
  IO.println msg

end torch.metrics
