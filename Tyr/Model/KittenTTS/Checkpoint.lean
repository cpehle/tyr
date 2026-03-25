/-
  Tyr/Model/KittenTTS/Checkpoint.lean

  Provider-backed typed schema for Kokoro / KittenTTS checkpoints.
  The provider is driven from a checked-in schema snapshot so elaboration does
  not depend on a machine-local checkpoint path.
-/
import Tyr.SafeTensors

open torch

namespace torch.kittentts

safetensors_type_provider "Tyr/Model/KittenTTS/kokoro_v1.schema.json" as KokoroCheckpoint

end torch.kittentts
