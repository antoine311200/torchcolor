from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from torch.nn import Module

from .color import TextColor, Color

@dataclass
class ModuleColor:
    color_name: TextColor = None
    color_descr: TextColor = None


class ColorStrategy(ABC):
    _registry = {}

    @abstractmethod
    def get_color(self, module: Module, params: dict) -> ModuleColor:
        """Return the appropriate color for the module based on some properties given by the strategy

        Args:
            module (Module): A Pytorch module
            params (dict): A dictionary with special parameters of the tree module node

        Returns:
            str: Color of the module
        """
        pass

    @classmethod
    def get_strategy(cls, key, *args, **kwargs):
        """
        Retrieve a color strategy instance based on its string key.

        Args:
            key (str): The key name of the strategy.
            *args: Positional arguments for the strategy's constructor.
            **kwargs: Keyword arguments for the strategy's constructor.

        Returns:
            ColorStrategy: An instance of the corresponding strategy.

        Raises:
            ValueError: If the key is not registered.
        """
        if key not in cls._registry:
            raise ValueError(f"Strategy '{key}' is not registered. Available strategies: {cls.available_strategies()}")
        return cls._registry[key](*args, **kwargs)

    @classmethod
    def register(cls, key):
        """Decorator to register a strategy with a specific string key."""
        def decorator(strategy_class):
            cls._registry[key] = strategy_class
            return strategy_class
        return decorator

    @classmethod
    def create(cls, key, *args, **kwargs):
        """Factory method to create an instance of a registered strategy by key."""
        if key not in cls._registry:
            raise ValueError(f"Strategy '{key}' is not registered.")
        return cls._registry[key](*args, **kwargs)

    @classmethod
    def available_strategies(cls):
        """Return a list of all available strategy keys."""
        return list(cls._registry.keys())


@ColorStrategy.register("trainable")
class TrainableColorStrategy(ColorStrategy):
    def get_color(self, module, config):
        params = list(module.parameters(recurse=True))
        if not params:
            return ModuleColor()
        elif all(not p.requires_grad for p in params):
            return ModuleColor(color_name=TextColor("red"))
        elif all(p.requires_grad for p in params):
            if config.is_leaf:
                return ModuleColor(color_name=TextColor("green"), color_descr=TextColor("black", "bright magenta"))
            else:
                return ModuleColor(color_name=TextColor("green"), color_descr=TextColor((45, 125, 201)))
        return ModuleColor(color_name=TextColor("yellow"), color_descr=TextColor((150, 100, 50)) if not config.is_root else None)

class LayerColorStrategy(ColorStrategy):
    def get_color(self, module):
        pass

class ConstantColorStrategy(ColorStrategy):
    def __init__(self, color: Union[str, tuple[int]] = ""):
        super().__init__()
        self.color = color

    def get_color(self, module, config):
        return ModuleColor(color_name=TextColor(self.color))