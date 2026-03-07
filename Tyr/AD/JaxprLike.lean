import Tyr.AD.JaxprLike.Core
import Tyr.AD.JaxprLike.KStmtNames
import Tyr.AD.JaxprLike.VertexOrder
import Tyr.AD.JaxprLike.Validate
import Tyr.AD.JaxprLike.FromFnBody
import Tyr.AD.JaxprLike.FromKStmt
import Tyr.AD.JaxprLike.LowerFnBody
import Tyr.AD.JaxprLike.LowerKStmt
import Tyr.AD.JaxprLike.Rules
import Tyr.AD.JaxprLike.RuleRegistry
import Tyr.AD.JaxprLike.RuleCheck
import Tyr.AD.JaxprLike.Extract
import Tyr.AD.JaxprLike.RulePackKStmt
import Tyr.AD.JaxprLike.Pipeline

/-!
# Tyr.AD.JaxprLike

Umbrella import for the LeanJaxpr-like IR layer used by elimination-based AD.
-/
