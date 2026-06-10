# Tyr Model Server

This is a separate experimental package that keeps the Cap'n Proto model-server
work out of the root `tyr` package graph.

The package depends on:

- `../..` for `Tyr`
- `../../../capnproto-lean` for Cap'n Proto RPC

That local-sibling dependency is intentional. It lets the server evolve against
the local `capnproto-lean` checkout without breaking the main `tyr` CI.

## What Exists

- `capnp/model_gateway.capnp`: a narrow text-generation RPC surface
- `TyrModelServer.Client`: minimal typed client wrapper over Cap'n Proto RPC
- `TyrModelServer.Protocol`: payload builders plus Qwen-style chat rendering
- `TyrModelServer.Qwen36`: a Qwen3.6 text server on top of `Tyr.Model.Qwen36`
- `lake exe smoke`: transport smoke test with a mock server
- `lake exe tyr_model_server`: real Qwen3.6 server executable

## Build

```bash
cd /Users/pehle/dev/tyr/dev/model-server
lake build
lake exe smoke
```

## Run

```bash
cd /Users/pehle/dev/tyr/dev/model-server
lake exe tyr_model_server --source Qwen/Qwen3.6-35B-A3B --device auto
```

Default address:

```text
unix:/tmp/tyr-qwen36.sock
```

## Client

The package also exposes a small typed client wrapper for downstream code:

```lean
import TyrModelServer

open TyrModelServer

def sample : IO Unit :=
  withModelGatewayConnection "unix:/tmp/tyr-qwen36.sock" fun conn => do
    let info ← conn.info
    let reply ← conn.generateUserText "Explain Lean elaboration in one sentence."
    IO.println s!"connected to {info.modelId}"
    IO.println reply.text
```

## Lean-Pi-Mono Mapping

The intended first integration path is:

1. `Sigma.AI.Context` in `lean-pi-mono` becomes `Array ChatMessage`.
2. `ModelGatewayCapability.stream` initially targets unary `generate`.
3. Streaming can be added later either with:
   - a token/event cursor capability, or
   - a sink callback capability for incremental deltas.

The current surface is intentionally narrow so the first end-to-end integration
can focus on loading, prompt formatting, and transport correctness.
