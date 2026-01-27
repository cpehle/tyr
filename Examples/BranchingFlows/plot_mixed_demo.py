#!/usr/bin/env python3
"""Plot outputs from Examples/BranchingFlows/MixedTrainDemo.lean.

Usage:
  python3 Examples/BranchingFlows/plot_mixed_demo.py

Expects the following files in the current working directory:
  examples_branching_mixed_pred.pt
  examples_branching_mixed_anchor.pt
  examples_branching_mixed_tokens.pt
  examples_branching_mixed_mask.pt
"""

import torch
import matplotlib.pyplot as plt

PRED_PATH = "examples_branching_mixed_pred.pt"
ANCHOR_PATH = "examples_branching_mixed_anchor.pt"
TOK_PATH = "examples_branching_mixed_tokens.pt"
MASK_PATH = "examples_branching_mixed_mask.pt"


def _load(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def main() -> None:
    pred = _load(PRED_PATH)
    anchor = _load(ANCHOR_PATH)
    tokens = _load(TOK_PATH)
    mask = _load(MASK_PATH)

    # Flatten batch/sequence dimensions
    pred2 = pred.reshape(-1, 2)
    anchor2 = anchor.reshape(-1, 2)
    tokens2 = tokens.reshape(-1)
    mask2 = mask.reshape(-1) > 0.5

    pred2 = pred2[mask2]
    anchor2 = anchor2[mask2]
    tokens2 = tokens2[mask2]

    fig, ax = plt.subplots(figsize=(5, 5))
    sc1 = ax.scatter(
        anchor2[:, 0],
        anchor2[:, 1],
        c=tokens2,
        cmap="viridis",
        s=14,
        alpha=0.5,
        label="anchor",
    )
    ax.scatter(
        pred2[:, 0],
        pred2[:, 1],
        c=tokens2,
        cmap="viridis",
        s=14,
        alpha=0.7,
        marker="x",
        label="pred",
    )
    ax.set_aspect("equal", "box")
    ax.set_title("BranchingFlows mixed demo")
    ax.legend(loc="best")
    fig.colorbar(sc1, ax=ax, label="token")
    fig.tight_layout()
    fig.savefig("examples_branching_mixed_plot.png", dpi=160)
    print("Wrote examples_branching_mixed_plot.png")


if __name__ == "__main__":
    main()
