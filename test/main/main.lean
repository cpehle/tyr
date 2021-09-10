import Tyr

def main : IO Unit := do
  let x <- torch.randn #[5,5]
  let y <- torch.rand #[5,5]
  let M := torch.zeros #[5,5]
  let b := torch.ones #[5]
  let z := (x + M)
  let a := (x - x)
  M[1].print
  (torch.ones #[5,5]).print
  z.print
  IO.println "------------------------"
  x.print
  (torch.linear x M).print
  (torch.affine x M b).print
  ((torch.zeros #[4,5,10]).slice (dim := 1) (start := 2)).print
  IO.println "------------------------"
  (torch.nn.softmax (torch.ones #[2,3])).print
  pure ()