#!/usr/bin/env python3
"""Synchronized PyTorch references for BF16 cross-entropy training."""

import argparse
import json
import torch
import torch.nn.functional as F


CASES = {
    "b1_s768_v3072": (768, {"training", "batch-sweep", "qwen3tts-talker"}, {"training", "latency"}),
    "b2_s768_v3072": (1536, {"training", "batch-sweep", "qwen3tts-talker"}, {"training", "throughput"}),
    "b4_s768_v3072": (3072, {"training", "batch-sweep", "qwen3tts-talker"}, {"training", "throughput", "primary"}),
    "b8_s768_v3072": (6144, {"training", "batch-sweep", "qwen3tts-talker"}, {"training", "throughput", "saturation"}),
    "micro_r128_v3072": (128, {"micro"}, {"training", "regression"}),
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
        raise ValueError("benchmark selection matched no loss cases")
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


def loss_with_aux(logits, targets):
    losses = F.cross_entropy(logits.float(), targets, reduction="none")
    return losses.mean(), losses


grad_and_value = torch.func.grad_and_value(loss_with_aux, argnums=0, has_aux=True)


def functional_training(logits, targets):
    grad, (mean_loss, losses) = grad_and_value(logits, targets)
    return losses, mean_loss, grad


def training_into(losses_out, mean_out, grad_out, logits, targets):
    losses, mean_loss, grad = functional_training(logits, targets)
    losses_out.copy_(losses)
    mean_out.copy_(mean_loss)
    grad_out.copy_(grad)
    return losses_out, mean_out, grad_out


def correctness(actual, expected):
    losses_ok = torch.allclose(actual[0].float(), expected[0].float(), rtol=2e-3, atol=2e-3)
    mean_ok = torch.allclose(actual[1].float(), expected[1].float(), rtol=2e-3, atol=2e-3)
    grad_ok = torch.allclose(actual[2], expected[2], rtol=3e-2, atol=3e-2)
    return bool(losses_ok and mean_ok and grad_ok)


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
        "workItemsPerIteration": work, "workItemUnit": "logits",
        "throughputItemsPerSecondP10": work * 1000.0 / p90,
        "throughputItemsPerSecondP50": work * 1000.0 / p50,
        "throughputItemsPerSecondP90": work * 1000.0 / p10,
        "torchVersion": torch.__version__, "cudaVersion": torch.version.cuda,
    }


def run_case(args, label, rows):
    logits = torch.randn((rows, 3072), device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, 3072, (rows,), device="cuda", dtype=torch.int64)
    expected = functional_training(logits, targets)
    losses_out = torch.empty((rows,), device="cuda", dtype=torch.float32)
    mean_out = torch.empty((), device="cuda", dtype=torch.float32)
    grad_out = torch.empty_like(logits)

    compiled = torch.compile(training_into, fullgraph=True)
    compiled(losses_out, mean_out, grad_out, logits, targets)
    torch.cuda.synchronize()

    logits_grad = logits.detach().requires_grad_(True)
    autograd_actual = [None]

    def autograd_step():
        losses = F.cross_entropy(logits_grad.float(), targets, reduction="none")
        mean_loss = losses.mean()
        grad = torch.autograd.grad(mean_loss, logits_grad)[0]
        autograd_actual[0] = (losses, mean_loss, grad)
        return autograd_actual[0]

    autograd_step()
    torch.cuda.synchronize()

    work = float(rows * 3072)
    case_id = f"cross_entropy_training_{label}"
    specs = [
        ("torch_eager", "torch_func_grad_and_value_cross_entropy",
         "full_training_operation", False,
         lambda: functional_training(logits, targets)),
        ("torch_eager_autograd", "torch_cross_entropy_actual_autograd",
         "forward_graph_plus_autograd_backward", False, autograd_step),
        ("torch_compile", "torch_compile_inductor_cross_entropy_training_into",
         "compiled_full_training_operation", True,
         lambda: compiled(losses_out, mean_out, grad_out, logits, targets)),
    ]
    result = []
    for backend, route, scope, alloc_free, fn in specs:
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
