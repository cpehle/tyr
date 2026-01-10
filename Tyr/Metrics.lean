/-
  Tyr/Metrics.lean

  Evaluation metrics for neural network training.
  Provides accuracy, precision, recall, F1 score, and other common metrics.
-/
import Tyr.Torch

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

end torch.metrics
