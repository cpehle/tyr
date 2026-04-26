/- Examples/NanoChat/Generator/KVCache.lean

  Compatibility shim: the canonical KV-cache module is now `Tyr.Inference.KVCache`
  so that any model in `Tyr/Model/*` can use it without depending on Examples/.
  This file is kept as a thin re-export so existing `import
  Examples.NanoChat.Generator.KVCache` calls keep working. New code should
  import `Tyr.Inference.KVCache` directly. -/
import Tyr.Inference.KVCache
