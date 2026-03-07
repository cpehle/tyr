import Tests.TestDiffEqUnderdampedOrderParity

unsafe def main : IO Unit := do
  Tests.DiffEqUnderdampedOrderParity.run
  IO.println "TestDiffEqUnderdampedOrderParity: ok"
