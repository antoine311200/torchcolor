import torch
from torch import nn

from torchcolor.printer import Printer
from torchcolor.strategy import GradientChangeStrategy
from torchcolor.snap import Snapshot

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.ModuleList(
            [nn.Linear(3, 3) for _ in range(10)]
        )
        for param in self.sequential[1].parameters(): param.requires_grad = False

def simulate_training_step(model, noise_std=0.1):
    for param in model.parameters():
        if param.requires_grad:
            random_noise = torch.randn_like(param.grad) * noise_std if param.grad is not None else torch.randn_like(param) * noise_std
            param.grad = random_noise


def grad_norm(module):
    if not list(module.parameters(recurse=True)):
        return 0
    return sum(p.grad.norm().item() for p in module.parameters() if p.grad is not None)


if __name__ == "__main__":
    model = SmallModel()
    snapshot = Snapshot(model, grad_norm)
    printer = Printer(strategy=GradientChangeStrategy())

    simulate_training_step(model)
    snapshot.capture("snap1")
    # print(snapshot.states)

    simulate_training_step(model)
    snapshot.capture("snap2")
    # print(snapshot.states)

    printer.print(model, snapshot=snapshot)