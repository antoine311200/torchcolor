from .strategy import ColorStrategy
from .color import colorize

# Function for adding indent from the pytorch codebase in /torch/nn/modules/module.py
def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def summarize_repeated_modules(lines):
    """
    Group repeated submodules into a single summary line if they have the same color and type.
    """
    if len(lines) == 0: return []
    # previous_key, previous_module_str, previous_color = lines[0]
    grouped_lines = []
    previous_key, previous_module_str, previous_color = None, None, None
    count = 0
    start_index = None

    for i, (key, mod_str, color) in enumerate(lines):
        if mod_str == previous_module_str and color == previous_color and (previous_key == key or key.isdigit()):
            if count == 0: start_index = i - 1
            count += 1
        else:
            if count > 0:
                # Summarize the group
                grouped_lines.pop()
                grouped_lines.append(
                    (f"{start_index}-{i-1}", f"{count + 1} x {mod_str}", previous_color)
                )
            grouped_lines.append((key, mod_str, color))
            count = 0
        previous_module_str = mod_str
        previous_color = color
        previous_key = key

    # # Handle the last group
    if count > 0:
        grouped_lines.pop()
        grouped_lines.append(
            (f"{start_index}-{len(lines)-1}", f"{count + 1} x {mod_str}", previous_color)
        )

    return grouped_lines

class Printer:

    def __init__(self, strategy: ColorStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ColorStrategy):
        """Change the strategy dynamically."""
        self.strategy = strategy

    def print(self, module):
        print(self.repr_module(module))

    def repr_module(self, parent_module, indent=2):
        """
        Recursively print the module with the chosen color strategy.
        """
        extra_lines = []
        extra_repr = parent_module.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []
        for key, module in parent_module._modules.items():
            if module is None:
                continue
            color = self.strategy.get_color(module)
            mod_str = self.repr_module(module, indent=indent + 2)
            child_lines.append((key, mod_str, color))

        summarized_lines = summarize_repeated_modules(child_lines)

        child_lines_formatted = []
        for key, mod_str, color in summarized_lines:
            colored_key = colorize(f"({key}):", text_color=color)
            child_lines_formatted.append(_addindent(f"{colored_key} {mod_str}", 2))

        lines = extra_lines + child_lines_formatted
        main_str = parent_module._get_name()
        if lines:
            if len(extra_lines) == 1 and not child_lines_formatted:
                main_str += "(" + extra_lines[0] + ")"
            else:
                main_str += "(\n  " + "\n  ".join(lines) + "\n)"

        return main_str