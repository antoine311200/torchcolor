from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from torch.nn import Module

from .style import TextStyle, FunctionalStyle, LAYER_SPLITTER, KeyType, DelimiterType, AnyType
from .gradient import Gradient

@dataclass
class ModuleStyle:
    """Dataclass for styling a module representative string

    Each module is usually printed as (name): layer(extra_information)
    As such, we provide a different style for each of these 3 blocks
    """
    name_style: Union[TextStyle, FunctionalStyle] = None
    layer_style: Union[TextStyle, FunctionalStyle] = None
    extra_style: Union[TextStyle, FunctionalStyle] = None


class ColorStrategy(ABC):
    """Styling strategy that allocate the corresponding module style to a given module based on registered pattern"""
    _registry = {}

    @abstractmethod
    def get_style(self, module: Module, params: dict) -> ModuleStyle:
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
class TrainableStrategy(ColorStrategy):
    """Styling strategy that handles trainable, non-trainable and mixed trainable layers/modules"""
    def get_style(self, module, config):
        params = list(module.parameters(recurse=True))
        if not params:
            return ModuleStyle()
        elif all(not p.requires_grad for p in params):
            return ModuleStyle(name_style=TextStyle("red"))
        elif all(p.requires_grad for p in params):
            return ModuleStyle(name_style=TextStyle("green"))
        return ModuleStyle(name_style=TextStyle("yellow")if not config.is_root else None)



class LayerColorStrategy(ColorStrategy):
    def get_style(self, module):
        pass



class ConstantColorStrategy(ColorStrategy):
    def __init__(self, color: Union[str, tuple[int]] = ""):
        super().__init__()
        self.color = color

    def get_style(self, module, config):
        return ModuleStyle(name_style=TextStyle(self.color, Gradient("warm_sunset"), double_underline=True, italic=True))