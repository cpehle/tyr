import Tyr.MctxDag.Base
import Tyr.MctxDag.Tree
import Tyr.MctxDag.QTransforms
import Tyr.MctxDag.ActionSelection
import Tyr.MctxDag.Search
import Tyr.MctxDag.Policies

/-!
# Tyr.MctxDag

`Tyr.MctxDag` is the DAG-oriented counterpart to `Tyr.Mctx`.
It re-exports search components specialized for graph/DAG-style expansion and reuse.

## Major Components

- DAG base/tree representations.
- Q-transforms and action-selection utilities.
- Search and policy orchestration over DAG state spaces.

## Scope

Use this module when your search problem benefits from DAG sharing semantics
rather than strict tree expansion.
-/
