from abc import ABC, abstractmethod
from torch.nn import Module


class ColorStrategy(ABC):
    @abstractmethod
    def get_color(self, module: Module) -> str:
        """Return the appropriate color for the module based on some properties given by the strategy

        Args:
            module (Module): A Pytorch module

        Returns:
            str: Color of the module
        """
        pass


class TrainableColorStrategy(ColorStrategy):
    def get_color(self, module):
        params = list(module.parameters(recurse=True))
        if not params:
            return ""
        elif all(not p.requires_grad for p in params):
            return "red"
        elif all(p.requires_grad for p in params):
            return "green"
        return "yellow"

class LayerColorStrategy(ColorStrategy):
    def get_color(self, module):
        pass