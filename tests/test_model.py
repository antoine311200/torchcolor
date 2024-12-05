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
        self.trainable_more_sequential = nn.ModuleList(
            [nn.Linear(3, 3) for _ in range(4)]
        )

        self.trainable_sequential = nn.ModuleList(
            [nn.Linear(5, 10)] + [nn.Linear(10, 10) for _ in range(10)] +
            [nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)) for _ in range(2)]
        )
        for param in self.trainable_sequential[4].parameters():
            param.requires_grad = False
        self.trainable_sequential[11][0].weight.requires_grad = False
        self.trainable_sequential[12][0].weight.requires_grad = False

        self.non_trainable_layer = nn.Linear(10, 10)
        for param in self.non_trainable_layer.parameters():
            param.requires_grad = False

        self.mixed_layer = nn.Sequential(
            nn.Linear(10, 10),  # Trainable
            nn.Linear(10, 10)  # Make this non-trainable
        )
        for param in self.mixed_layer[1].parameters():
            param.requires_grad = False

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.ModuleList(
            [nn.Linear(3, 3) for _ in range(5)]
        )
        for param in self.sequential[2].parameters(): param.requires_grad = False

from torchcolor.printer import Printer, ModuleStyle
from torchcolor.strategy import ColorStrategy, ConstantColorStrategy
from torchcolor.gradient import Gradient
from torchcolor.color import Color
from torchcolor.style import TextStyle, FunctionalStyle, LAYER_SPLITTER, DelimiterType, AnyType, KeyType

class SmallStrategy(ColorStrategy):
    def process(self, module, config):
        params = list(module.parameters(recurse=True))
        if not params:
            return ModuleStyle()
        elif all(not p.requires_grad for p in params):
            return ModuleStyle(extra_style=TextStyle("red"))
        elif all(p.requires_grad for p in params):
            return ModuleStyle(extra_style=TextStyle("green"))
        return ModuleStyle(name_style=TextStyle("yellow"))

@ColorStrategy.register("custom")
class CustomStrategy(ColorStrategy):
    def process(self, name, module, config):
        params = list(module.parameters(recurse=True))
        if not params:
            return ModuleStyle()
        elif all(not p.requires_grad for p in params):
            if config.is_leaf:
                return ModuleStyle(name_style=TextStyle(bg_style=Gradient("warm_sunset")), extra_style=FunctionalStyle(splitter=LAYER_SPLITTER, styles={
                    KeyType: TextStyle((45, 124, 85)),
                    AnyType: TextStyle(underline=True),
                    DelimiterType: TextStyle(italic=True),
                    bool: TextStyle((25, 120, 230), italic=True)
                }))
            else:
                return ModuleStyle(name_style=TextStyle("red"))
        elif all(p.requires_grad for p in params):
            if config.is_leaf:
                return ModuleStyle(
                    name_style=TextStyle(bg_style=Color("blue")),
                    layer_style=TextStyle("bright magenta", double_underline=True),
                    extra_style=FunctionalStyle(splitter=LAYER_SPLITTER, styles={
                        KeyType: TextStyle(Gradient("warm_sunset")),
                        AnyType: TextStyle(underline=True),
                        DelimiterType: TextStyle(italic=True),
                        bool: TextStyle((180, 25, 120), italic=True)
                    })
                )
            else:
                return ModuleStyle(
                    name_style=TextStyle("green"),
                    layer_style=TextStyle((45, 125, 201)),
                )
        return ModuleStyle(name_style=TextStyle(bg_style=Gradient("rainbow")), layer_style=TextStyle((150, 100, 50)) if not config.is_root else None)


if __name__ == "__main__":
    # model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    # model = SimpleModel()

    print("Torch module loaded.\n")
    model = DiverseModel()
    printer = Printer(strategy="custom")
    printer.print(model)

    printer.set_strategy(ConstantColorStrategy((40, 80, 20)))
    printer.print(model)

    printer.set_strategy("trainable")
    printer.print(model, legend=True)