import Examples.GPU.RunRMSNorm

/-- Lake executable entrypoint for the fused residual + RMSNorm demo. -/
def main (args : List String) : IO UInt32 :=
  Examples.GPU.RunRMSNorm.main args
