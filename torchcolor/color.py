from dataclasses import dataclass
from typing import Union

foreground_colors = {
    "reset": 0,
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "gray": 90,
    "bright red": 91,
    "bright green": 92,
    "bright yellow": 93,
    "bright blue": 94,
    "bright magenta": 95,
    "bright cyan": 96,
    "bright white": 97,
}

background_colors = {
    "black": 40,
    "red": 41,
    "green": 42,
    "yellow": 43,
    "blue": 44,
    "magenta": 45,
    "cyan": 46,
    "white": 47,
    "gray": 100,
    "bright red": 101,
    "bright green": 102,
    "bright yellow": 103,
    "bright blue": 104,
    "bright magenta": 105,
    "bright cyan": 106,
    "bright white": 107
}

def hex_to_rgb(hex_color):
    """Convert a hex color string (e.g., "#RRGGBB") to an RGB tuple."""
    hex_color = hex_color.lstrip("#").lower()
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

@dataclass
class Color:
    value: Union[str, tuple[int, int, int]]

    def to_rgb(self) -> tuple[int, int, int]:
        if isinstance(self.value, tuple):
            if len(self.value) != 3:
                raise ValueError("RGB color must have three components (R, G, B).")
            if any(int(component) != component or component > 255 or component < 0 for component in self.value):
                raise ValueError("RGB components must be integers in the range [0, 255]")
            return self.value
        elif isinstance(self.value, str):
            if self.value in foreground_colors or self.value in background_colors:
                raise ValueError("Cannot convert named 4-bit colors to RGB directly.")
            if not self.value.startswith("#"):
                self.value = f"#{self.value}"
            return hex_to_rgb(self.value)
        else:
            raise TypeError("Unsupported color format.")


    def to_hex(self) -> str:
        if isinstance(self.value, str):
            pass

    def to_ansi(self, is_background: bool = False) -> str:
        if self.value is None:
            return ""
        elif isinstance(self.value, tuple):
            red, green, blue = self.value
            return f"\033[{48 if is_background else 38};2;{red};{green};{blue}m"
        elif isinstance(self.value, str):
            color_map = background_colors if is_background else foreground_colors
            if self.value in color_map:
                return f"\033[{color_map[self.value]}m"
            else:
                red, green, blue = self.to_rgb()
                return f"\033[{48 if is_background else 38};2;{red};{green};{blue}m"
        else:
            raise TypeError("Unsupported color format.")

reset_color = Color("reset")


@dataclass
class TextColor:
    fg_color: Color = None
    bg_color: Color = None

    def __init__(self, fg_color: Union[Color, str, tuple[int, int, int]] = None, bg_color: Union[Color, str, tuple[int, int, int]] = None):
        if not isinstance(fg_color, Color):
            self.fg_color = Color(fg_color)
        if not isinstance(bg_color, Color):
            self.bg_color = Color(bg_color)

    def apply(self, text: str) -> str:
        return reset_color.to_ansi() + self.fg_color.to_ansi() + self.bg_color.to_ansi(is_background=True) + text + reset_color.to_ansi()

def colorize(
    text: str,
    text_color: Union[Color, str, tuple[int, int, int]] = None,
    bg_color: Union[Color, str, tuple[int, int, int]] = None
) -> str:
    """Colorize the console text with a foreground and background color

    Args:
        text (str): String text to colorize
        text_color (str, optional): Color of the text foreground. Defaults to None.
        bg_color (str, optional): Color of the text background. Defaults to None.

    Returns:
        str: Formatted console ANSI escape code for colorizing text
    """
    foreground_color = text_color if isinstance(text_color, Color) else Color(text_color)
    background_color = bg_color if isinstance(bg_color, Color) else Color(bg_color)

    return TextColor(foreground_color, background_color).apply(text)

def print_color(text:str, text_color: str = None, bg_color: str = None) -> str:
    print(colorize(text, text_color=text_color, bg_color=bg_color))