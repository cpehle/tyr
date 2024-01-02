import Tyr



def print (s : String) := do
  let stdin ← IO.getStdin.toIO
  IO.print "\r"
  stdin.flush

def main := do
  let a ← torch.randn #[2,5]
  let b ← torch.randn #[2,5]
  let c ← torch.randn #[5,2]
  IO.println $ a + b

  IO.println $ torch.allclose a a
  
  IO.println $ (torch.reshape a #[5,2]) + c
  IO.println $ torch.permute a #[1,0]

  IO.println $ a.permute #[1,0]
