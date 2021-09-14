import Tyr

def input : torch.T #[8,8] := torch.ones #[8,8] (requires_grad := true)
def grad_output : torch.T #[8,8] := torch.ones #[8,8] 
def param : torch.T #[8,8] := torch.ones #[8,8] (requires_grad := true)
def output : torch.T #[8,8] := torch.nn.relu input

#eval torch.autograd.pullback (torch.nn.relu ∘ (torch.linear param)) input grad_output
#eval torch.randn #[10,10]