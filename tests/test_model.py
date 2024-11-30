from torchcolor import Printer
from torchcolor.strategy import ConstantColorStrategy

from transformers import AutoModelForSequenceClassification

import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32, 10)
        self.dropout = nn.Dropout()

    def forward(self, x):
        pass

class DiverseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable_layer = nn.Linear(10, 10)  # Trainable by default
        self.trainable_sequential = nn.ModuleList(
            [nn.Linear(5, 10)] + [nn.Linear(10, 10) for _ in range(10)] +
            [nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)) for _ in range(2)]
        )
        self.non_trainable_layer = nn.Linear(10, 10)
        for param in self.non_trainable_layer.parameters():
            param.requires_grad = False

        for param in self.trainable_sequential[4].parameters():
            param.requires_grad = False
        self.mixed_layer = nn.Sequential(
            nn.Linear(10, 10),  # Trainable
            nn.Linear(10, 10)  # Make this non-trainable
        )
        for param in self.mixed_layer[1].parameters():
            param.requires_grad = False

if __name__ == "__main__":
    # model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    # model = SimpleModel()
    model = DiverseModel()
    printer = Printer(strategy="trainable")
    printer.print(model)

    printer.set_strategy(ConstantColorStrategy((124, 210, 79)))
    printer.print(model)
    # for param in model.parameters():
    #     param.requires_grad = False
    # printer.print(model)