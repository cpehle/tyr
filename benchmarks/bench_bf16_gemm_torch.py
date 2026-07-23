#!/usr/bin/env python3
"""Synchronized PyTorch references for configurable BF16 GEMM suites."""
import argparse
import json

import torch


CASES = {
    "tiny_m256_n256_k64": (
        256, 256, 64, {"quick", "micro"}, {"launch-bound", "forward"},
    ),
}
TRAINING_BATCH_POINTS = (
    ("b1", 768, {"latency"}),
    ("b2", 1536, {"throughput"}),
    ("b4", 3072, {"throughput", "primary"}),
    ("b8", 6144, {"throughput", "saturation"}),
)
for label, tokens, scale_tags in TRAINING_BATCH_POINTS:
    model_profiles = {"qwen3tts-talker"}
    if "primary" in scale_tags:
        model_profiles.add("qwen3tts-talker-primary")
    projection_tags = {"training", "projection"} | scale_tags
    mlp_tags = {"training", "mlp"} | scale_tags
    CASES.update({
        f"square_h1024_{label}_s768_fwd_dx": (
            tokens, 1024, 1024,
            {"gb10-realistic", "model-shapes", "batch-sweep",
             "training-triplets", "projection-triplets"} | model_profiles,
            projection_tags | {"q-output", "forward", "activation-gradient"},
        ),
        f"square_h1024_{label}_s768_dw": (
            1024, 1024, tokens,
            {"model-shapes", "training-triplets", "weight-grad-sweep",
             "projection-triplets"} | model_profiles,
            projection_tags | {"q-output", "weight-gradient"},
        ),
        f"qwen3tts_talker_kv_{label}_s768_fwd": (
            tokens, 128, 1024,
            {"model-shapes", "projection-triplets"} | model_profiles,
            projection_tags | {"kv", "forward"},
        ),
        f"qwen3tts_talker_kv_{label}_s768_dx": (
            tokens, 1024, 128,
            {"model-shapes", "projection-triplets"} | model_profiles,
            projection_tags | {"kv", "activation-gradient"},
        ),
        f"qwen3tts_talker_kv_{label}_s768_dw": (
            128, 1024, tokens,
            {"model-shapes", "projection-triplets"} | model_profiles,
            projection_tags | {"kv", "weight-gradient"},
        ),
        f"qwen3tts_talker_mlp_up_{label}_s768_fwd_down_dx": (
            tokens, 2048, 1024,
            {"model-shapes", "mlp-triplets"} | model_profiles,
            mlp_tags | {"up-gate-forward", "down-activation-gradient"},
        ),
        f"qwen3tts_talker_mlp_down_{label}_s768_fwd_up_dx": (
            tokens, 1024, 2048,
            {"model-shapes", "mlp-triplets"} | model_profiles,
            mlp_tags | {"down-forward", "up-gate-activation-gradient"},
        ),
        f"qwen3tts_talker_mlp_up_{label}_s768_dw": (
            2048, 1024, tokens,
            {"model-shapes", "mlp-triplets"} | model_profiles,
            mlp_tags | {"up-gate-weight-gradient"},
        ),
        f"qwen3tts_talker_mlp_down_{label}_s768_dw": (
            1024, 2048, tokens,
            {"model-shapes", "mlp-triplets"} | model_profiles,
            mlp_tags | {"down-weight-gradient"},
        ),
    })
DEFAULT_PROFILE = "gb10-realistic"


def split_selections(values):
    return {item.strip() for value in values or [] for item in value.split(",") if item.strip()}


def select_cases(args):
    if args.m is not None or args.n is not None or args.k is not None:
        return [("custom", (args.m if args.m is not None else 256, args.n if args.n is not None else 256, args.k if args.k is not None else 64))]
    requested = split_selections(args.case)
    tags = split_selections(args.tag)
    unknown = requested.difference(CASES)
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {sorted(unknown)}")
    selected = []
    for name, (m, n, k, profiles, case_tags) in CASES.items():
        if args.profile not in profiles:
            continue
        if requested and name not in requested:
            continue
        if tags and not tags.intersection(case_tags):
            continue
        selected.append((name, (m, n, k)))
    if not selected:
        raise ValueError("benchmark selection matched no cases")
    return selected


def percentile(xs, pct):
    xs = sorted(xs)
    return xs[((len(xs) - 1) * pct) // 100]


def measure(fn, warmup, iters, repeats):
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


def functional(a, b):
    return torch.matmul(a, b.transpose(0, 1))


def eager_into(out, a, b):
    return torch.mm(a, b.transpose(0, 1), out=out)


def compiled_into(out, a, b):
    out.copy_(functional(a, b))
    return out


def tolerances(k):
    atol = 1.0 if k >= 3072 else 0.5 if k >= 1536 else 5e-2
    return 5e-2, atol


def error_stats(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return float(diff.mean().item()), float(diff.max().item())


def make_row(args, case, backend, route, samples, ok, scope, allocation_free, flops, rtol, atol, stats):
    p10 = percentile(samples, 10)
    p50 = percentile(samples, 50)
    p90 = percentile(samples, 90)
    return {
        "event": "summary", "schemaVersion": 1, "runId": args.run_id,
        "caseId": case, "backend": backend, "routeActual": route,
        "timer": "cuda_event", "completionFence": "cudaEventSynchronize(stop)",
        "timingScope": scope, "warmAllocationFree": allocation_free,
        "compileSetupExcluded": backend == "torch_compile",
        "warmup": args.warmup, "iters": args.iters, "repeats": args.repeats,
        "correctnessOk": ok, "latencyMsP10": p10, "latencyMsP50": p50,
        "rtol": rtol, "atol": atol, "mae": stats[0], "maxError": stats[1],
        "latencyMsP90": p90, "workItemsPerIteration": flops,
        "workItemUnit": "FLOP",
        "throughputItemsPerSecondP10": flops * 1000.0 / p90,
        "throughputItemsPerSecondP50": flops * 1000.0 / p50,
        "throughputItemsPerSecondP90": flops * 1000.0 / p10,
        "torchVersion": torch.__version__, "cudaVersion": torch.version.cuda,
    }


def run_case(args, case_label, m, n, k):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    expected = torch.matmul(a.float(), b.float().transpose(0, 1)).to(torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(compiled_into, fullgraph=True)
    compiled(out, a, b)
    torch.cuda.synchronize()
    rtol, atol = tolerances(k)
    compiled_ok = bool(torch.allclose(out, expected, rtol=rtol, atol=atol))
    compiled_stats = error_stats(out, expected)
    eager_out = functional(a, b)
    eager_ok = bool(torch.allclose(eager_out, expected, rtol=rtol, atol=atol))
    eager_stats = error_stats(eager_out, expected)
    eager_into(out, a, b)
    torch.cuda.synchronize()
    eager_into_ok = bool(torch.allclose(out, expected, rtol=rtol, atol=atol))
    eager_into_stats = error_stats(out, expected)
    eager = measure(lambda: functional(a, b), args.warmup, args.iters, args.repeats)
    eager_kernel = measure(lambda: eager_into(out, a, b), args.warmup, args.iters, args.repeats)
    comp = measure(lambda: compiled(out, a, b), args.warmup, args.iters, args.repeats)
    case = f"bf16_gemm_{case_label}_{m}x{n}x{k}"
    flops = float(2 * m * n * k)
    return [
        make_row(args, case, "torch_eager", "torch_matmul_abt", eager, eager_ok, "end_to_end", False, flops, rtol, atol, eager_stats),
        make_row(args, case, "torch_eager_kernel", "torch_mm_out_abt", eager_kernel, eager_into_ok, "kernel_only", True, flops, rtol, atol, eager_into_stats),
        make_row(args, case, "torch_compile", "torch_compile_inductor", comp, compiled_ok, "kernel_plus_output_copy", True, flops, rtol, atol, compiled_stats),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--jsonl-out", required=True)
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--case", action="append")
    p.add_argument("--tag", action="append")
    p.add_argument("--m", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--k", type=int)
    args = p.parse_args()
    torch.manual_seed(0)
    try:
        selected = select_cases(args)
    except ValueError as exc:
        p.error(str(exc))
    if any(m <= 0 or n <= 0 or m % 64 or n % 64 for _, (m, n, _) in selected):
        p.error("m and n must be positive multiples of 64")
    if any(k != 64 and (k < 128 or k % 64) for _, (_, _, k) in selected):
        p.error("k must be 64 or a multiple of 64 greater than or equal to 128")
    rows = []
    for case_label, (m, n, k) in selected:
        rows.extend(run_case(args, case_label, m, n, k))
    with open(args.jsonl_out, "a", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, sort_keys=True)
            f.write(line + "\n")
            print(line)
    return 0 if all(row["correctnessOk"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
