import Tyr.Mctx.Base
import Tyr.Mctx.Tree
import Tyr.Mctx.SeqHalving
import Tyr.Mctx.QTransforms
import Tyr.Mctx.ActionSelection
import Tyr.Mctx.Search
import Tyr.Mctx.Policies
import Tyr.Mctx.Batched

/-!
# Tyr.Mctx

`Tyr.Mctx` is the umbrella import for Tyr's Monte Carlo tree search stack.
It re-exports the core types and algorithms used for search-based planning/inference.

## Major Components

- Base data model and bookkeeping (`Mctx.Base`, `Mctx.Tree`).
- Search refinements (`SeqHalving`, Q-value transforms, action selection).
- Policy/search orchestration (`Mctx.Search`, `Mctx.Policies`).
- Batched search helpers (`Mctx.Batched`).

## Scope

Use this module as the standard entrypoint for tree-search workflows.
Import submodules directly when you want a narrower dependency surface.
-/
