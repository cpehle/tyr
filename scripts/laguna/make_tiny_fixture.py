#!/usr/bin/env python
"""Generate the tiny-Laguna HF parity fixture for cross-engine validation.

Builds a small random Laguna checkpoint in the real HF layout (BF16
safetensors, exact `modeling_laguna.py` state_dict names) plus fp32 reference
outputs computed with the official reference implementation in
`dev/laguna_reference/`, and writes everything to
`Tests/fixtures/laguna/tiny/`:

  config.json           HF LagunaConfig (tiny), via cfg.to_json_file
  model.safetensors     BF16 weights, single shard, HF checkpoint layout
  reference.safetensors fp32 reference tensors (logits / hidden states)
  reference.json        input ids, greedy generated ids, and meta

Weight determinism: every state_dict tensor is refilled from a dedicated
seeded torch.Generator in sorted-key order (2D/3D ~ N(0, 0.02), norm weights
~ 1 + 0.02N, e_score_correction_bias ~ 0.01N so the bias path is exercised),
then rounded to BF16. The BF16-rounded values (upcast back to fp32) are the
canonical weights: all references are computed from them, so a fresh reload of
the checkpoint reproduces logits_a bit-exactly.

References are computed on CPU in fp32 with attn_implementation="eager".

Run from the repo root:
    .venv-gpu/bin/python scripts/laguna/make_tiny_fixture.py
"""

import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_DIR = REPO_ROOT / "dev" / "laguna_reference"
OUT_DIR = REPO_ROOT / "Tests" / "fixtures" / "laguna" / "tiny"

# Import the reference implementation as a package (modeling_laguna.py uses a
# relative import of configuration_laguna).
pkg = types.ModuleType("laguna_reference")
pkg.__path__ = [str(REF_DIR)]
sys.modules["laguna_reference"] = pkg

from laguna_reference.configuration_laguna import LagunaConfig  # noqa: E402
from laguna_reference.modeling_laguna import LagunaForCausalLM  # noqa: E402

WEIGHT_SEED = 1234
B_IDS_SEED = 5678
INIT_STD = 0.02  # matches config.initializer_range
NORM_JITTER = 0.02
BIAS_STD = 0.01

INPUT_A = [2, 37, 101, 456, 1000, 5, 9]
GEN_STEPS = 16


def build_config() -> LagunaConfig:
    rope_parameters = {
        "full_attention": {
            "rope_theta": 500000.0,
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
            "beta_slow": 1.0,
            "beta_fast": 32.0,
            "attention_factor": 1.3465735902799727,
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
    }
    cfg = LagunaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_heads_per_layer=[4, 6, 6, 6],
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=1024,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        norm_topk_prob=True,
        moe_routed_scaling_factor=2.5,
        mlp_only_layers=[0],
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        sliding_window=8,
        # Per-head output gating: matches the real poolside/Laguna-S-2.1-NVFP4
        # checkpoint (g_proj [num_heads, hidden]) and the tyr model, which
        # implements per-head gating only. HF's default True would be
        # per-element (g_proj [num_heads*head_dim, hidden]).
        gating="per-head",
        rms_norm_eps=1e-6,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        bos_token_id=2,
        eos_token_id=[2, 24],
        pad_token_id=9,
        rope_parameters=rope_parameters,
    )
    cfg._attn_implementation = "eager"
    return cfg


def refill_weights_(model: torch.nn.Module) -> None:
    """Deterministically refill every state_dict tensor (sorted-key order)."""
    g = torch.Generator(device="cpu").manual_seed(WEIGHT_SEED)
    with torch.no_grad():
        for name, t in model.state_dict().items():
            if name.endswith("e_score_correction_bias"):
                # HF zero-inits this; use small nonzero values so a missing
                # bias-add in the engine under test cannot hide.
                t.copy_(BIAS_STD * torch.randn(t.shape, generator=g, dtype=torch.float32))
            elif t.ndim == 1:
                # RMSNorm weights (HF inits to ones; jitter so the weight
                # multiply is exercised).
                t.copy_(1.0 + NORM_JITTER * torch.randn(t.shape, generator=g, dtype=torch.float32))
            else:
                t.copy_(INIT_STD * torch.randn(t.shape, generator=g, dtype=torch.float32))


def roundtrip_bf16_(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Round all weights through BF16; return the BF16 state_dict to save."""
    sd = model.state_dict()
    sd_bf16 = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in sd.items()}
    with torch.no_grad():
        for name, t in sd_bf16.items():
            sd[name].copy_(t.to(torch.float32))
    return sd_bf16


@torch.no_grad()
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)  # construction-time init (overwritten below anyway)

    cfg = build_config()
    model = LagunaForCausalLM(cfg)
    model.eval()

    refill_weights_(model)
    sd_bf16 = roundtrip_bf16_(model)

    # ---- checkpoint artifacts ------------------------------------------------
    save_file(sd_bf16, str(OUT_DIR / "model.safetensors"), metadata={"format": "pt"})
    cfg.to_json_file(str(OUT_DIR / "config.json"))

    # ---- inputs ---------------------------------------------------------------
    ids_a = torch.tensor([INPUT_A], dtype=torch.long)
    g_b = torch.Generator(device="cpu").manual_seed(B_IDS_SEED)
    input_b = torch.randint(0, cfg.vocab_size, (20,), generator=g_b).tolist()
    ids_b = torch.tensor([input_b], dtype=torch.long)

    # ---- references (fp32, CPU, eager attention) ------------------------------
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        model.model.layers[0].register_forward_hook(
            lambda m, i, o: captured.__setitem__("hidden_a_l0", o.detach().clone())
        ),
        model.model.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("hidden_a_final", o.detach().clone())
        ),
    ]
    out_a = model(input_ids=ids_a, use_cache=False)
    for h in hooks:
        h.remove()
    logits_a = out_a.logits[0].float().contiguous()  # [7, vocab]

    logits_b = model(input_ids=ids_b, use_cache=False).logits[0].float().contiguous()  # [20, vocab]

    gen_ids: list[int] = []
    past = None
    cur = ids_a
    gen_logits_last = None
    for _ in range(GEN_STEPS):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        step_logits = out.logits[0, -1].float()
        nxt = int(step_logits.argmax().item())
        gen_ids.append(nxt)
        gen_logits_last = step_logits
        cur = torch.tensor([[nxt]], dtype=torch.long)

    reference = {
        "logits_a": logits_a,
        "logits_b": logits_b,
        "gen_logits_last": gen_logits_last.contiguous(),
        "hidden_a_l0": captured["hidden_a_l0"][0].float().contiguous(),
        "hidden_a_final": captured["hidden_a_final"][0].float().contiguous(),
    }
    save_file(reference, str(OUT_DIR / "reference.safetensors"), metadata={"format": "pt"})

    # ---- self-consistency: fresh reload must reproduce logits_a ---------------
    cfg2 = LagunaConfig.from_pretrained(str(OUT_DIR))
    cfg2._attn_implementation = "eager"
    # NB: transformers v5 from_pretrained preserves the checkpoint dtype (bf16)
    # by default; references are fp32, so force the upcast for an exact check.
    model2 = LagunaForCausalLM.from_pretrained(str(OUT_DIR), config=cfg2, dtype=torch.float32)
    model2.eval()
    with torch.no_grad():
        logits_a2 = model2(input_ids=ids_a, use_cache=False).logits[0].float()
    reload_max_diff = (logits_a2 - logits_a).abs().max().item()

    sd2 = model2.state_dict()
    weight_mismatches = [
        k for k in sd_bf16 if not torch.equal(sd2[k].to(torch.bfloat16), sd_bf16[k])
    ]

    # ---- reference.json -------------------------------------------------------
    meta = {
        "seed_weights": WEIGHT_SEED,
        "seed_input_b": B_IDS_SEED,
        "weight_init": (
            f"all state_dict tensors refilled via torch.Generator(seed={WEIGHT_SEED}) in "
            f"sorted-key order: ndim>=2 ~ N(0, {INIT_STD}), 1-D norms ~ 1+{NORM_JITTER}N, "
            f"e_score_correction_bias ~ {BIAS_STD}N; then rounded to bfloat16"
        ),
        "dtype_checkpoint": "bfloat16",
        "dtype_reference": "float32",
        "device": "cpu",
        "attn_implementation": "eager",
        "gen": "greedy argmax over full vocab, KV cache enabled, prompt=input_ids_a",
        "gen_steps": GEN_STEPS,
        "shapes": {k: list(v.shape) for k, v in reference.items()},
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "reload_logits_a_max_abs_diff": reload_max_diff,
        "reload_weight_mismatches": weight_mismatches,
    }
    ref_json = {
        "input_ids_a": INPUT_A,
        "input_ids_b": input_b,
        "gen_ids": gen_ids,
        "meta": meta,
    }
    with open(OUT_DIR / "reference.json", "w") as f:
        json.dump(ref_json, f, indent=2)

    # ---- summary ----------------------------------------------------------------
    print(f"wrote {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}: {p.stat().st_size} bytes")
    print(f"input_ids_a: {INPUT_A}")
    print(f"input_ids_b: {input_b}")
    print(f"gen_ids: {gen_ids}")
    print(f"reload logits_a max abs diff: {reload_max_diff}")
    print(f"reload weight mismatches: {weight_mismatches}")
    if reload_max_diff != 0.0 or weight_mismatches:
        sys.exit("FAIL: fixture is not self-consistent")
    print("OK: fixture is self-consistent")


if __name__ == "__main__":
    main()
