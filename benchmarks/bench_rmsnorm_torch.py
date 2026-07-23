#!/usr/bin/env python3
"""Synchronized PyTorch references for BF16 residual RMSNorm training."""

import argparse
import json
import torch


CASES = {
    "micro_64": (64, {"micro"}, {"forward", "regression"}),
    "b1_s768": (768, {"training", "batch-sweep", "qwen3tts-talker"}, {"training", "latency"}),
    "b2_s768": (1536, {"training", "batch-sweep", "qwen3tts-talker"}, {"training", "throughput"}),
    "b4_s768": (3072, {"training", "batch-sweep", "qwen3tts-talker"}, {"training", "throughput", "primary"}),
    "b8_s768": (6144, {"training", "batch-sweep", "qwen3tts-talker"}, {"training", "throughput", "saturation"}),
}
DEFAULT_PROFILE = "training"


def selections(values):
    return {x.strip() for value in values or [] for x in value.split(",") if x.strip()}


def selected_cases(args):
    wanted, tags = selections(args.case), selections(args.tag)
    unknown = wanted.difference(CASES)
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {sorted(unknown)}")
    result = []
    for name, (rows, profiles, case_tags) in CASES.items():
        if args.profile not in profiles:
            continue
        if wanted and name not in wanted:
            continue
        if tags and not tags.intersection(case_tags):
            continue
        result.append((name, rows))
    if not result:
        raise ValueError("benchmark selection matched no RMSNorm cases")
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
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iters)
    return samples


def forward(x, residual, weight):
    saved = x + residual
    saved_f = saved.float()
    inv = torch.rsqrt(saved_f.square().mean(dim=-1) + 1.0e-6)
    out = (saved_f * inv[:, None] * weight.float()).to(torch.bfloat16)
    return out, saved, inv


def backward_input(grad_out, grad_saved, saved, weight, inv):
    saved_f = saved.float()
    weighted = grad_out.float() * weight.float()
    dot = (weighted * saved_f).sum(dim=-1)
    correction = dot * inv * inv * inv * (1.0 / 1024.0)
    return (
        weighted * inv[:, None]
        - saved_f * correction[:, None]
        + grad_saved.float()
    ).to(torch.bfloat16)


def backward_weight(grad_out, saved, inv):
    return (grad_out.float() * saved.float() * inv[:, None]).sum(dim=0)


def training(x, residual, weight, grad_out, grad_saved):
    out, saved, inv = forward(x, residual, weight)
    dx = backward_input(grad_out, grad_saved, saved, weight, inv)
    dw = backward_weight(grad_out, saved, inv)
    return out, saved, inv, dx, dw


def forward_into(out, saved_out, inv_out, x, residual, weight):
    out_value, saved_value, inv_value = forward(x, residual, weight)
    out.copy_(out_value)
    saved_out.copy_(saved_value)
    inv_out.copy_(inv_value)
    return out, saved_out, inv_out


def backward_input_into(dx, grad_out, grad_saved, saved, weight, inv):
    dx.copy_(backward_input(grad_out, grad_saved, saved, weight, inv))
    return dx


def backward_weight_into(dw, grad_out, saved, inv):
    dw.copy_(backward_weight(grad_out, saved, inv))
    return dw


def training_into(out, saved_out, inv_out, dx, dw, x, residual, weight,
                  grad_out, grad_saved):
    values = training(x, residual, weight, grad_out, grad_saved)
    for target, value in zip((out, saved_out, inv_out, dx, dw), values):
        target.copy_(value)
    return out, saved_out, inv_out, dx, dw


def as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def correctness(actual, expected):
    actual, expected = as_tuple(actual), as_tuple(expected)
    if len(actual) != len(expected):
        return False
    for value, reference in zip(actual, expected):
        if reference.dtype == torch.float32 and reference.numel() == 1024:
            if not torch.allclose(value.float(), reference, rtol=3e-3, atol=0.25):
                return False
        elif reference.dtype == torch.float32:
            if not torch.allclose(value.float(), reference, rtol=2e-4, atol=2e-4):
                return False
        elif not torch.allclose(value, reference, rtol=3e-2, atol=3e-2):
            return False
    return True


def make_row(args, case_id, backend, route, scope, allocation_free,
             work, samples, ok):
    p10, p50, p90 = (percentile(samples, p) for p in (10, 50, 90))
    return {
        "event": "summary", "schemaVersion": 1, "runId": args.run_id,
        "caseId": case_id, "backend": backend, "routeActual": route,
        "timer": "cuda_event", "completionFence": "cudaEventSynchronize(stop)",
        "timingScope": scope, "warmAllocationFree": allocation_free,
        "compileSetupExcluded": backend == "torch_compile",
        "warmup": args.warmup, "iters": args.iters, "repeats": args.repeats,
        "correctnessOk": ok, "latencyMsP10": p10,
        "latencyMsP50": p50, "latencyMsP90": p90,
        "workItemsPerIteration": work, "workItemUnit": "elements",
        "throughputItemsPerSecondP10": work * 1000.0 / p90,
        "throughputItemsPerSecondP50": work * 1000.0 / p50,
        "throughputItemsPerSecondP90": work * 1000.0 / p10,
        "torchVersion": torch.__version__, "cudaVersion": torch.version.cuda,
    }


def run_case(args, label, rows):
    shape = (rows, 1024)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((1024,), device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    grad_saved = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected_fwd = forward(x, residual, weight)
    expected_dx = backward_input(grad_out, grad_saved, expected_fwd[1], weight, expected_fwd[2])
    expected_dw = backward_weight(grad_out, expected_fwd[1], expected_fwd[2])
    expected_training = expected_fwd + (expected_dx, expected_dw)

    out, saved_out = torch.empty_like(x), torch.empty_like(x)
    inv_out = torch.empty((rows,), device="cuda", dtype=torch.float32)
    dx, dw = torch.empty_like(x), torch.empty((1024,), device="cuda", dtype=torch.float32)

    compiled_fwd = torch.compile(forward_into, fullgraph=True)
    compiled_dx = torch.compile(backward_input_into, fullgraph=True)
    compiled_dw = torch.compile(backward_weight_into, fullgraph=True)
    compiled_training = torch.compile(training_into, fullgraph=True)

    compiled_fwd(out, saved_out, inv_out, x, residual, weight)
    compiled_dx(dx, grad_out, grad_saved, saved_out, weight, inv_out)
    compiled_dw(dw, grad_out, saved_out, inv_out)
    compiled_training(
        out, saved_out, inv_out, dx, dw, x, residual, weight,
        grad_out, grad_saved,
    )
    torch.cuda.synchronize()

    x_grad = x.detach().requires_grad_(True)
    residual_grad = residual.detach().requires_grad_(True)
    weight_grad = weight.detach().requires_grad_(True)
    autograd_actual = [None]

    def autograd_step():
        out_value, saved_value, _ = forward(x_grad, residual_grad, weight_grad)
        grads = torch.autograd.grad(
            (out_value, saved_value),
            (x_grad, residual_grad, weight_grad),
            (grad_out, grad_saved),
        )
        autograd_actual[0] = (grads[0], grads[1], grads[2].float())
        return autograd_actual[0]

    autograd_step()
    torch.cuda.synchronize()
    expected_autograd = (expected_dx, expected_dx, expected_dw)

    elements = float(rows * 1024)
    specs = [
        (f"rmsnorm_training_fwd_{label}", "torch_eager",
         "torch_explicit_saved_state_forward", "saved_state_forward", False,
         lambda: forward(x, residual, weight), expected_fwd, elements),
        (f"rmsnorm_training_fwd_{label}", "torch_compile",
         "torch_compile_inductor_saved_state_forward_into",
         "compiled_saved_state_forward", True,
         lambda: compiled_fwd(out, saved_out, inv_out, x, residual, weight),
         expected_fwd, elements),
        (f"rmsnorm_training_bwd_input_{label}", "torch_eager",
         "torch_explicit_input_residual_vjp", "input_and_residual_vjp", False,
         lambda: backward_input(grad_out, grad_saved, expected_fwd[1], weight, expected_fwd[2]),
         expected_dx, elements),
        (f"rmsnorm_training_bwd_input_{label}", "torch_compile",
         "torch_compile_inductor_input_residual_vjp_into",
         "compiled_input_and_residual_vjp", True,
         lambda: compiled_dx(dx, grad_out, grad_saved, saved_out, weight, inv_out),
         expected_dx, elements),
        (f"rmsnorm_training_bwd_weight_{label}", "torch_eager",
         "torch_explicit_weight_gradient", "weight_gradient", False,
         lambda: backward_weight(grad_out, expected_fwd[1], expected_fwd[2]),
         expected_dw, elements),
        (f"rmsnorm_training_bwd_weight_{label}", "torch_compile",
         "torch_compile_inductor_weight_gradient_into",
         "compiled_weight_gradient", True,
         lambda: compiled_dw(dw, grad_out, saved_out, inv_out),
         expected_dw, elements),
        (f"rmsnorm_training_{label}", "torch_eager",
         "torch_explicit_forward_plus_vjps", "full_training_operation", False,
         lambda: training(x, residual, weight, grad_out, grad_saved),
         expected_training, 3.0 * elements),
        (f"rmsnorm_training_{label}", "torch_eager_autograd",
         "torch_autograd_forward_plus_actual_vjps",
         "forward_graph_plus_autograd_backward", False,
         autograd_step, expected_autograd, 3.0 * elements),
        (f"rmsnorm_training_{label}", "torch_compile",
         "torch_compile_inductor_full_training_into",
         "compiled_full_training_operation", True,
         lambda: compiled_training(
             out, saved_out, inv_out, dx, dw, x, residual, weight,
             grad_out, grad_saved,
         ), expected_training, 3.0 * elements),
    ]

    result = []
    for case_id, backend, route, scope, alloc_free, fn, expected, work in specs:
        actual = fn()
        torch.cuda.synchronize()
        ok = correctness(actual, expected)
        samples = measure(fn, args.warmup, args.iters, args.repeats)
        actual = fn()
        torch.cuda.synchronize()
        ok = ok and correctness(actual, expected)
        result.append(make_row(
            args, case_id, backend, route, scope, alloc_free,
            work, samples, ok,
        ))
    return result


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
