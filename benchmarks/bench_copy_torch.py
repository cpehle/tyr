#!/usr/bin/env python3
"""CUDA-event PyTorch reference for Tyr's fixed 64x64 FP32 copy kernel."""
import argparse
import json
import statistics
import torch


def measure(fn, warmup: int, iters: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iters)
    return samples


def percentile(samples: list[float], pct: int) -> float:
    ordered = sorted(samples)
    return ordered[((len(ordered) - 1) * pct) // 100]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--jsonl-out", required=True)
    args = p.parse_args()
    if args.iters <= 0 or args.repeats <= 0:
        p.error("--iters and --repeats must be positive")

    torch.manual_seed(1001)
    src = torch.rand((1, 1, 64, 64), device="cuda", dtype=torch.float32)
    dst = torch.zeros_like(src)

    def copy_into(out: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        out.copy_(value)
        return out

    compiled = torch.compile(copy_into, fullgraph=True)
    compiled(dst, src)  # compile outside every measured region
    torch.cuda.synchronize()
    correct = bool(torch.equal(dst, src))
    samples = measure(lambda: compiled(dst, src), args.warmup, args.iters, args.repeats)
    row = {
        "event": "summary",
        "schemaVersion": 1,
        "runId": args.run_id,
        "caseId": "copy_f32_64x64",
        "backend": "torch_compile",
        "routeActual": "torch_compile_inductor",
        "timer": "cuda_event", "completionFence": "cudaEventSynchronize(stop)",
        "warmAllocationFree": True,
        "compileSetupExcluded": True,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "correctnessOk": correct,
        "latencyMsP10": percentile(samples, 10),
        "latencyMsP50": percentile(samples, 50),
        "latencyMsP90": percentile(samples, 90),
        "torchVersion": torch.__version__,
        "cudaVersion": torch.version.cuda,
    }
    with open(args.jsonl_out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
