from typing import Union

color_4bits_fg = {
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

color_4bits_bg = {
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


def colorize(text: str, text_color: Union[str, tuple[int]] = None, bg_color: Union[str, tuple[int]] = None) -> str:
    """Colorize the console text with a foreground and background color

    Args:
        text (str): String text to colorize
        text_color (str, optional): Color of the text foreground. Defaults to None.
        bg_color (str, optional): Color of the text background. Defaults to None.

    Returns:
        str: Formatted console ANSI escape code for colorizing text
    """
    foreground_color = ""
    background_color = ""

    if isinstance(text_color, tuple):
        if len(text_color) != 3:
            raise TypeError("text_color of type tuple does not have length 3 for (red, green blue) components.")
        red, green, blue = text_color
        foreground_color = f"\033[38;2;{red};{green};{blue}m"
    elif isinstance(text_color, str) and len(text_color) > 0:
        if text_color not in color_4bits_fg.keys():
            if text_color[0] != '#': text_color = '#'+text_color
            red, green, blue = hex_to_rgb(text_color)
            foreground_color = f"\033[38;2;{red};{green};{blue}m"
        else:
            foreground_color = f"\033[{color_4bits_fg[text_color]}m"

    if isinstance(bg_color, tuple):
        if len(bg_color) != 3:
            raise TypeError("bg_color of type tuple does not have length 3 for (red, green blue) components.")
        red, green, blue = bg_color
        background_color = f"\033[48;2;{red};{green};{blue}m"
    elif isinstance(bg_color, str) and len(bg_color) > 0:
        if bg_color not in color_4bits_bg.keys():
            if bg_color[0] != '#': bg_color = '#'+bg_color
            red, green, blue = hex_to_rgb(bg_color)
            background_color = f"\033[48;2;{red};{green};{blue}m"
        else:
            background_color = f"\033[{color_4bits_bg[bg_color]}m"

    return (
        foreground_color +
        background_color +
        text +
        "\033[0m"
    )

def print_color(text:str, text_color: str = None, bg_color: str = None) -> str:
    print(colorize(text, text_color=text_color, bg_color=bg_color))