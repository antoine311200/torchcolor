from typing import Union
from .style import colorize, TextStyle
from .palette import Palette

def print_color(text:str, text_color: str = None, bg_color: str = None) -> str:
    print(colorize(text, text_color=text_color, bg_color=bg_color))

def print_more(*args: Union[str, TextStyle], **kwargs):
    results = []
    sep = kwargs.get("sep", " ")
    buffer_string = ""
    for arg in args:
        if isinstance(arg, str):
            if len(buffer_string) > 0:
                results.append(buffer_string)
            buffer_string = arg
        elif isinstance(arg, TextStyle):
            results.append(arg.apply(buffer_string))
            buffer_string = ""
        else:
            raise TypeError(f"Argument {arg} is neither a string or a TextColor instance")
    print(sep.join(results))