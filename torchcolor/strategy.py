from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

from torch.nn import Module

from .style import TextStyle, FunctionalStyle, LAYER_SPLITTER, KeyType, DelimiterType, AnyType

@dataclass
class ModuleStyle:
    name_style: Union[TextStyle, FunctionalStyle] = None
    layer_style: Union[TextStyle, FunctionalStyle] = None
    extra_style: Union[TextStyle, FunctionalStyle] = None


class ColorStrategy(ABC):
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
class TrainableColorStrategy(ColorStrategy):
    def get_style(self, module, config):
        params = list(module.parameters(recurse=True))
        if not params:
            return ModuleStyle()
        elif all(not p.requires_grad for p in params):
            if config.is_leaf:
                return ModuleStyle(name_style=TextStyle("red"), extra_style=FunctionalStyle(splitter=LAYER_SPLITTER, styles={
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
                    name_style=TextStyle("green"),
                    layer_style=TextStyle("black", "bright magenta"),
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
        return ModuleStyle(name_style=TextStyle("yellow"), layer_style=TextStyle((150, 100, 50)) if not config.is_root else None)

class LayerColorStrategy(ColorStrategy):
    def get_style(self, module):
        pass


from .gradient import Gradient

class ConstantColorStrategy(ColorStrategy):
    def __init__(self, color: Union[str, tuple[int]] = ""):
        super().__init__()
        self.color = color

    def get_style(self, module, config):
        return ModuleStyle(name_style=TextStyle(self.color, Gradient("warm_sunset"), double_underline=True, italic=True))