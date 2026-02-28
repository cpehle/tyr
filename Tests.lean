import Tests.Test
import Tests.TestDiffusion
import Tests.TestDataLoader
import Tests.TestModdedGPT
import Tests.TestCheckpoint
import Tests.TestModelIO
import Tests.TestPipeline
import Tests.TestAutoGrad
import Tests.TestGPUAutoGrad
import Tests.TestModularNorm
import Tests.TestMctxImport
import Tests.TestMctxSeqHalving
import Tests.TestMctxQTransforms
import Tests.TestMctxPolicies
import Tests.TestMctxTree
import Tests.TestMctx
import Tests.TestMctxBatched
import Tests.TestMctxDag
import Tests.TestDiffEq
import Tests.TestDiffEqAdjoint
import Tests.TestDiffEqAdjointCore
import Tests.TestGPUDSL
import Tests.TestGPUKernels
import Tests.TestNanoChatTokens
import Tests.TestNanoChatTasks
import Tests.TestQwen3ASR
import Tests.TestQwen3Model
import Tests.TestQwen35Model
import Tests.TestQwen35Multimodal
import Tests.TestQwen35Pretrained
import Tests.TestSafeTensorsTypeProvider
-- Note: `Tests.TestQwen3TTS` depends on in-progress Qwen3TTS weight-loading
-- modules and is temporarily excluded from the default test runner.

-- Note: `Tests.TestBranchingFlows` is currently routed through
-- `TestsExperimental.lean` until `Tyr/Model/BranchingFlows.lean` is stabilized.
