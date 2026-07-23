#!/usr/bin/env python3
"""Synchronized PyTorch references for mixed-precision AdamW updates."""

import argparse
import json
import torch

CASES = {
    "qkv_weight_n1048576": (1048576, {"qwen3tts-talker-primary", "training"}, {"attention", "matrix", "primary"}),
    "kv_weight_n131072": (131072, {"qwen3tts-talker-primary", "training"}, {"attention", "gqa", "small"}),
    "mlp_weight_n2097152": (2097152, {"qwen3tts-talker-primary", "training"}, {"mlp", "matrix", "primary"}),
    "embedding_n3145728": (3145728, {"qwen3tts-talker-primary", "training"}, {"embedding", "adam", "large"}),
    "micro_n4096": (4096, {"micro"}, {"regression"}),
}
DEFAULT_PROFILE = "qwen3tts-talker-primary"
LR, BETA1, BETA2, EPS, WEIGHT_DECAY = 3e-4, 0.9, 0.95, 1e-8, 0.1
INV_BIAS1, INV_BIAS2 = 1 / (1 - BETA1), 1 / (1 - BETA2)


def selections(values):
    return {x.strip() for value in values or [] for x in value.split(",") if x.strip()}


def selected_cases(args):
    wanted, tags = selections(args.case), selections(args.tag)
    unknown = wanted.difference(CASES)
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {sorted(unknown)}")
    result = []
    for name, (elements, profiles, case_tags) in CASES.items():
        if args.profile not in profiles or (wanted and name not in wanted):
            continue
        if tags and not tags.intersection(case_tags):
            continue
        result.append((name, elements))
    if not result:
        raise ValueError("benchmark selection matched no optimizer cases")
    return result


def update_into(master_out, model_out, m_out, v_out, master, grad, m, v):
    grad32 = grad.float()
    next_m = torch.lerp(grad32, m, BETA1)
    next_v = torch.lerp(grad32.square(), v, BETA2)
    update = (next_m * INV_BIAS1) / ((next_v * INV_BIAS2).sqrt() + EPS)
    next_master = master - LR * (update + WEIGHT_DECAY * master)
    master_out.copy_(next_master)
    model_out.copy_(next_master)
    m_out.copy_(next_m)
    v_out.copy_(next_v)
    return master_out, model_out, m_out, v_out


def percentile(samples, pct):
    samples = sorted(samples)
    return samples[((len(samples) - 1) * pct) // 100]


def measure(fn, warmup, iters, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iters)
    return samples


def correctness(actual, expected):
    tolerances = ((2e-5, 2e-5), (2e-3, 2e-3), (2e-5, 2e-5), (2e-5, 2e-5))
    return all(torch.allclose(a.float(), e.float(), rtol=r, atol=at)
               for a, e, (r, at) in zip(actual, expected, tolerances))


def make_row(args, case_id, backend, route, allocation_free, work, samples, ok):
    p10, p50, p90 = (percentile(samples, p) for p in (10, 50, 90))
    return {
        "event": "summary", "schemaVersion": 1, "runId": args.run_id,
        "caseId": case_id, "backend": backend, "routeActual": route,
        "timer": "cuda_event", "completionFence": "cudaEventSynchronize(stop)",
        "timingScope": "full_mixed_precision_optimizer_update",
        "warmAllocationFree": allocation_free,
        "compileSetupExcluded": backend == "torch_compile",
        "warmup": args.warmup, "iters": args.iters, "repeats": args.repeats,
        "correctnessOk": ok, "latencyMsP10": p10, "latencyMsP50": p50,
        "latencyMsP90": p90, "workItemsPerIteration": float(work),
        "workItemUnit": "parameters",
        "throughputItemsPerSecondP10": work * 1000 / p90,
        "throughputItemsPerSecondP50": work * 1000 / p50,
        "throughputItemsPerSecondP90": work * 1000 / p10,
        "torchVersion": torch.__version__, "cudaVersion": torch.version.cuda,
    }


def run_case(args, label, elements):
    master = torch.randn(elements, device="cuda", dtype=torch.float32)
    grad = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
    m, v = torch.zeros_like(master), torch.zeros_like(master)
    outputs = (torch.empty_like(master),
               torch.empty(elements, device="cuda", dtype=torch.bfloat16),
               torch.empty_like(master), torch.empty_like(master))
    expected_outputs = tuple(torch.empty_like(x) for x in outputs)
    expected = update_into(*expected_outputs, master, grad, m, v)
    compiled = torch.compile(update_into, fullgraph=True)
    compiled(*outputs, master, grad, m, v)
    torch.cuda.synchronize()  # compilation and first execution are excluded
    specs = [
        ("torch_eager", "torch_functional_adamw_update_into", False,
         lambda: update_into(*outputs, master, grad, m, v)),
        ("torch_compile", "torch_compile_inductor_adamw_update_into", True,
         lambda: compiled(*outputs, master, grad, m, v)),
    ]
    rows = []
    for backend, route, allocation_free, fn in specs:
        actual = fn()
        torch.cuda.synchronize()
        ok = correctness(actual, expected)
        samples = measure(fn, args.warmup, args.iters, args.repeats)
        actual = fn()
        torch.cuda.synchronize()
        ok = ok and correctness(actual, expected)
        rows.append(make_row(args, f"adamw_training_{label}", backend, route,
                             allocation_free, elements, samples, ok))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--jsonl-out", required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--case", action="append")
    parser.add_argument("--tag", action="append")
    args = parser.parse_args()
    torch.manual_seed(0)
    try:
        cases = selected_cases(args)
    except ValueError as error:
        parser.error(str(error))
    rows = [row for case in cases for row in run_case(args, *case)]
    with open(args.jsonl_out, "a", encoding="utf-8") as output:
        for row in rows:
            line = json.dumps(row, sort_keys=True)
            output.write(line + "\n")
            print(line)
    return 0 if all(row["correctnessOk"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
