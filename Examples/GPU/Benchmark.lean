/- Tyr CUDA adapter for the backend-neutral LeanBenchmark package. -/
import LeanBenchmark
import Tyr.Torch

namespace Examples.GPU.Benchmark

open torch

abbrev Config := LeanBenchmark.Config

def parseConfig := LeanBenchmark.parseConfig

private def cudaEventTimer : LeanBenchmark.EventTimer := {
  create := torch.cuda_event_create
  record := fun event stream => torch.cuda_event_record event stream
  synchronize := torch.cuda_event_synchronize
  elapsedMs := torch.cuda_event_elapsed_ms
  destroy := torch.cuda_event_destroy
  timerName := "cuda_event"
  completionFence := "cudaEventSynchronize(stop)"
}

/-- Time a preallocated action on stream with CUDA events. The generic runner
    synchronizes the stop event before reading elapsed time. -/
def timeCudaEvents (cfg : Config) (stream : UInt64)
    (action : IO Unit) : IO (Array Float) :=
  LeanBenchmark.timeEvents cudaEventTimer cfg stream action

def summaryJson (cfg : Config) (caseId backend route : String)
    (samples : Array Float) (correct : Bool)
    (timingScope : String := "kernel_only")
    (warmAllocationFree : Bool := true)
    (workItemsPerIteration : Option Float := none)
    (workItemUnit : Option String := none) : String :=
  LeanBenchmark.summaryJson cfg caseId backend route samples correct
    timingScope warmAllocationFree cudaEventTimer.timerName
    cudaEventTimer.completionFence workItemsPerIteration workItemUnit

def writeJsonl := LeanBenchmark.writeJsonl

end Examples.GPU.Benchmark
