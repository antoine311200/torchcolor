
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

def colorize(text:str, text_color: str = None, bg_color: str = None) -> str:
    """Colorize the console text with a foreground and background color

    Args:
        text (str): String text to colorize
        text_color (str, optional): Color of the text foreground. Defaults to None.
        bg_color (str, optional): Color of the text background. Defaults to None.

    Returns:
        str: Formatted console ANSI escape code for colorizing text
    """
    return (
        (f"\033[{color_4bits_fg[text_color]}m" if text_color else "") +
        (f"\033[{color_4bits_bg[bg_color]}m" if bg_color else "") +
        text +
        "\033[0m"
    )

def print_color(text:str, text_color: str = None, bg_color: str = None) -> str:
    print(colorize(text, text_color=text_color, bg_color=bg_color))