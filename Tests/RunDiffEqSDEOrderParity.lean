import Tests.TestDiffEqSDEOrderParity

unsafe def main : IO Unit := do
  Tests.DiffEqSDEOrderParity.run
  IO.println "TestDiffEqSDEOrderParity: ok"
