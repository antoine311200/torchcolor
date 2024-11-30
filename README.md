# Torchcolor

### Description

### Usage

#### Strategy: Trainable

```python
class DiverseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Trainable
        self.trainable_layer = nn.Linear(10, 10)

        # Trainable for all except the 4th module
        self.trainable_sequential = nn.ModuleList(
            [nn.Linear(5, 10)] + [nn.Linear(10, 10) for _ in range(10)] +
            [nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)) for _ in range(2)]
        )
        for param in self.trainable_sequential[4].parameters():
            param.requires_grad = False

        # Non trainable layer
        self.non_trainable_layer = nn.Linear(10, 10)
        for param in self.non_trainable_layer.parameters():
            param.requires_grad = False

        # Mixed trainable layer
        self.mixed_layer = nn.Sequential(
            nn.Linear(10, 10),  # Trainable
            nn.Linear(10, 10)  # Make this non-trainable
        )
        for param in self.mixed_layer[1].parameters():
            param.requires_grad = False

model = DiverseModel()
printer = Printer(strategy=TrainableColorStrategy())
printer.print(model)
```

![](https://private-user-images.githubusercontent.com/8627785/391143286-0470bf1c-03f5-45ff-89a8-2d7bf790117b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzI5MzMyNjgsIm5iZiI6MTczMjkzMjk2OCwicGF0aCI6Ii84NjI3Nzg1LzM5MTE0MzI4Ni0wNDcwYmYxYy0wM2Y1LTQ1ZmYtODlhOC0yZDdiZjc5MDExN2IucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MTEzMCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDExMzBUMDIxNjA4WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9MjZjMjAyYjViZGRkMTEwZjI1NTFjMDkyNTllODE0MDQ4MzM1NTBlNzBlOTk2NjBhMTBlODhlNjU3MjhkMDY5YSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.BoYyGgeNcoSBq53tryFoII55hvAs64gmVXwchSnqOJQ)