#!/usr/bin/env python3
"""Synchronized PyTorch references for configurable BF16 RoPE training suites."""

import argparse
import json
import torch


CASES = {
    "b1_h16_kv2_s768": (1, 768, 16, 2, {"training", "batch-sweep", "gqa", "qwen3tts-talker"}, {"training", "gqa", "latency"}),
    "b2_h16_kv2_s768": (2, 768, 16, 2, {"training", "batch-sweep", "gqa", "qwen3tts-talker"}, {"training", "gqa", "throughput"}),
    "b4_h16_kv2_s768": (4, 768, 16, 2, {"training", "batch-sweep", "gqa", "qwen3tts-talker"}, {"training", "gqa", "throughput", "primary"}),
    "b8_h16_kv2_s768": (8, 768, 16, 2, {"training", "batch-sweep", "gqa", "qwen3tts-talker"}, {"training", "gqa", "throughput", "saturation"}),
    "b1_h16_kv16_s768": (1, 768, 16, 16, {"equal-head", "mha-compat"}, {"training", "latency"}),
    "b2_h16_kv16_s768": (2, 768, 16, 16, {"equal-head", "mha-compat"}, {"training", "throughput"}),
    "b4_h16_kv16_s768": (4, 768, 16, 16, {"equal-head", "mha-compat"}, {"training", "throughput", "primary"}),
    "b8_h16_kv16_s768": (8, 768, 16, 16, {"equal-head", "mha-compat"}, {"training", "throughput", "saturation"}),
}
DEFAULT_PROFILE = "training"


def selections(values):
    return {x.strip() for value in values or [] for x in value.split(",") if x.strip()}


def selected_cases(args):
    wanted, wanted_tags = selections(args.case), selections(args.tag)
    unknown = wanted.difference(CASES)
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {sorted(unknown)}")
    result = []
    for name, spec in CASES.items():
        batch, seq_len, q_heads, kv_heads, profiles, tags = spec
        if args.profile not in profiles:
            continue
        if wanted and name not in wanted:
            continue
        if wanted_tags and not wanted_tags.intersection(tags):
            continue
        result.append((name, batch, seq_len, q_heads, kv_heads))
    if not result:
        raise ValueError("benchmark selection matched no RoPE cases")
    return result


def percentile(samples, pct):
    samples = sorted(samples)
    return samples[((len(samples) - 1) * pct) // 100]


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
        # The synchronized stop event proves all measured GPU work completed.
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iters)
    return samples


def rotary(x, sin, cos):
    x1, x2 = x[..., :32].float(), x[..., 32:].float()
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).to(torch.bfloat16)


def rotary_vjp(grad, sin, cos):
    dy1, dy2 = grad[..., :32].float(), grad[..., 32:].float()
    return torch.cat((dy1 * cos + dy2 * sin, dy2 * cos - dy1 * sin), -1).to(torch.bfloat16)


def fwd(q, k, sin, cos):
    return rotary(q, sin, cos), rotary(k, sin, cos)


def bwd(grad_q, grad_k, sin, cos):
    return rotary_vjp(grad_q, sin, cos), rotary_vjp(grad_k, sin, cos)


def training(q, k, grad_q, grad_k, sin, cos):
    return fwd(q, k, sin, cos) + bwd(grad_q, grad_k, sin, cos)


def fwd_into(q_out, k_out, q, k, sin, cos):
    q_value, k_value = fwd(q, k, sin, cos)
    q_out.copy_(q_value)
    k_out.copy_(k_value)
    return q_out, k_out


def bwd_into(dq_out, dk_out, grad_q, grad_k, sin, cos):
    dq_value, dk_value = bwd(grad_q, grad_k, sin, cos)
    dq_out.copy_(dq_value)
    dk_out.copy_(dk_value)
    return dq_out, dk_out


def training_into(q_out, k_out, dq_out, dk_out, q, k, grad_q, grad_k, sin, cos):
    q_value, k_value, dq_value, dk_value = training(q, k, grad_q, grad_k, sin, cos)
    q_out.copy_(q_value)
    k_out.copy_(k_value)
    dq_out.copy_(dq_value)
    dk_out.copy_(dk_value)
    return q_out, k_out, dq_out, dk_out


def correctness(actual, expected, tol=3e-2):
    ok = all(torch.allclose(a, e, rtol=tol, atol=tol) for a, e in zip(actual, expected))
    total, count, maximum = 0.0, 0, 0.0
    for value, reference in zip(actual, expected):
        diff = (value.float() - reference.float()).abs()
        total += float(diff.sum().item())
        count += diff.numel()
        maximum = max(maximum, float(diff.max().item()))
    return bool(ok), total / count, maximum


def summary(args, case_id, backend, route, timing_scope, allocation_free,
            work, shape, samples, ok, mae, max_error):
    p10, p50, p90 = (percentile(samples, p) for p in (10, 50, 90))
    batch, seq_len, q_heads, kv_heads = shape
    return {
        "event": "summary", "schemaVersion": 1, "runId": args.run_id,
        "caseId": case_id, "backend": backend, "routeActual": route,
        "timer": "cuda_event", "completionFence": "cudaEventSynchronize(stop)",
        "timingScope": timing_scope, "warmAllocationFree": allocation_free,
        "compileSetupExcluded": backend == "torch_compile",
        "warmup": args.warmup, "iters": args.iters, "repeats": args.repeats,
        "correctnessOk": ok, "rtol": 3e-2, "atol": 3e-2,
        "mae": mae, "maxError": max_error,
        "latencyMsP10": p10, "latencyMsP50": p50, "latencyMsP90": p90,
        "workItemsPerIteration": work, "workItemUnit": "rotated_pairs",
        "throughputItemsPerSecondP10": work * 1000.0 / p90,
        "throughputItemsPerSecondP50": work * 1000.0 / p50,
        "throughputItemsPerSecondP90": work * 1000.0 / p10,
        "batch": batch, "sequenceLength": seq_len, "qHeads": q_heads,
        "kvHeads": kv_heads, "headDimension": 64,
        "inputDtype": "bfloat16", "outputDtype": "bfloat16",
        "trigDtype": "float32", "torchVersion": torch.__version__,
        "cudaVersion": torch.version.cuda,
    }


def run_case(args, label, batch, seq_len, q_heads, kv_heads):
    shape = batch, seq_len, q_heads, kv_heads
    q_shape, k_shape = (batch, seq_len, q_heads, 64), (batch, seq_len, kv_heads, 64)
    q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(k_shape, device="cuda", dtype=torch.bfloat16)
    grad_q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    grad_k = torch.randn(k_shape, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64.0
    ))
    angles = positions[:, None] * inv_freq[None, :]
    sin, cos = angles.sin()[None, :, None, :], angles.cos()[None, :, None, :]

    expected_fwd = fwd(q, k, sin, cos)
    expected_bwd = bwd(grad_q, grad_k, sin, cos)
    expected_training = expected_fwd + expected_bwd
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    dq_out, dk_out = torch.empty_like(q), torch.empty_like(k)

    compiled_fwd = torch.compile(fwd_into, fullgraph=True)
    compiled_bwd = torch.compile(bwd_into, fullgraph=True)
    compiled_training = torch.compile(training_into, fullgraph=True)

    # Compilation, code generation, autotuning, and first launch all finish here.
    compiled_fwd(q_out, k_out, q, k, sin, cos)
    compiled_bwd(dq_out, dk_out, grad_q, grad_k, sin, cos)
    compiled_training(q_out, k_out, dq_out, dk_out, q, k, grad_q, grad_k, sin, cos)
    torch.cuda.synchronize()

    q_grad, k_grad = q.detach().requires_grad_(True), k.detach().requires_grad_(True)
    autograd_result = [None]

    def autograd_step():
        q_value, k_value = fwd(q_grad, k_grad, sin, cos)
        autograd_result[0] = torch.autograd.grad(
            (q_value, k_value), (q_grad, k_grad), (grad_q, grad_k)
        )
        return autograd_result[0]

    # First autograd graph construction/launch is also outside timed events.
    autograd_step()
    torch.cuda.synchronize()

    pair_work = float((q.numel() + k.numel()) // 2)
    specs = [
        (f"rotary_qk_fwd_{label}", "torch_eager",
         "torch_split_half_bf16_functional", "end_to_end_allocating_forward",
         False, lambda: fwd(q, k, sin, cos), expected_fwd, pair_work),
        (f"rotary_qk_fwd_{label}", "torch_compile",
         "torch_compile_inductor_forward_into",
         "compiled_forward_into_preallocated_outputs", True,
         lambda: compiled_fwd(q_out, k_out, q, k, sin, cos),
         expected_fwd, pair_work),
        (f"rotary_dqdk_bwd_{label}", "torch_eager",
         "torch_explicit_rope_vjp_functional", "end_to_end_allocating_backward",
         False, lambda: bwd(grad_q, grad_k, sin, cos), expected_bwd, pair_work),
        (f"rotary_dqdk_bwd_{label}", "torch_compile",
         "torch_compile_inductor_explicit_vjp_into",
         "compiled_backward_into_preallocated_outputs", True,
         lambda: compiled_bwd(dq_out, dk_out, grad_q, grad_k, sin, cos),
         expected_bwd, pair_work),
        (f"rotary_training_{label}", "torch_eager",
         "torch_explicit_forward_plus_vjp", "forward_plus_explicit_backward",
         False, lambda: training(q, k, grad_q, grad_k, sin, cos),
         expected_training, 2.0 * pair_work),
        (f"rotary_training_{label}", "torch_eager_autograd",
         "torch_autograd_forward_plus_actual_vjp",
         "forward_graph_plus_autograd_backward", False,
         autograd_step, expected_bwd, 2.0 * pair_work),
        (f"rotary_training_{label}", "torch_compile",
         "torch_compile_inductor_forward_plus_explicit_vjp_into",
         "compiled_forward_plus_backward_into_preallocated_outputs", True,
         lambda: compiled_training(
             q_out, k_out, dq_out, dk_out, q, k, grad_q, grad_k, sin, cos
         ), expected_training, 2.0 * pair_work),
    ]
    rows = []
    for case_id, backend, route, scope, alloc_free, fn, expected, work in specs:
        actual = fn()
        torch.cuda.synchronize()
        ok, mae, max_error = correctness(actual, expected)
        samples = measure(fn, args.warmup, args.iters, args.repeats)
        actual = fn()
        torch.cuda.synchronize()
        post_ok, post_mae, post_max = correctness(actual, expected)
        rows.append(summary(
            args, case_id, backend, route, scope, alloc_free, work, shape,
            samples, ok and post_ok, post_mae, post_max,
        ))
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
    rows = []
    for case in cases:
        rows.extend(run_case(args, *case))
    with open(args.jsonl_out, "a", encoding="utf-8") as output:
        for row in rows:
            line = json.dumps(row, sort_keys=True)
            output.write(line + "\n")
            print(line)
    return 0 if all(row["correctnessOk"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
