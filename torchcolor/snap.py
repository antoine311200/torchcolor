from typing import Callable
from collections import OrderedDict, defaultdict

from torch import nn

from .utils import iterate

class Snapshot:

    def __init__(self, model: nn.Module, snap_func: Callable):
        self.model = model
        self.snap_func = snap_func

        self.states = defaultdict(OrderedDict)

    def capture(self, name: str = None, recurse: bool = True):
        if not name:
            name = max(len(d) for d in self.states.values())

        if any(name in module_state for module_state in self.states.values()):
            raise NameError(f"the snapshot named `{name}` already exists.")

        if recurse:
            for module_name, module in self.model.named_modules():
                self.states[module_name][name] = self.snap_func(module)
        else:
            self.states[self.model._get_name()][name] = self.snap_func(self.model)

    def retro(self, module_name: str, capture_name: str = None, index: int = 1):
        if not capture_name:
            return self.states[module_name][iterate(reversed(self.states[module_name]), index)]
        return self.states[module_name][capture_name]
