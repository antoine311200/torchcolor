from dataclasses import dataclass
from collections import defaultdict
from typing import Union, Type, Literal
import re

from .color import Color, reset_color
from .gradient import Gradient

def clean_style(text: str) -> str:
    ansi_escape_regex = r'\033\[[0-9;]*m'
    return re.sub(ansi_escape_regex, '', text)

@dataclass
class TextStyle:
    """Text style dataclass with foreground and background colors/gradients and additional properties
    such as underline, italic and so on."""

    # May add support for underline style later [escape code 58]
    fg_style: Union[Gradient, Color, str, tuple[int, int, int]] = None
    bg_style: Union[Gradient, Color, str, tuple[int, int, int]] = None

    bold: bool = False
    italic: bool = False
    underline: bool = False
    double_underline: bool = False
    crossed: bool = False
    darken: bool = False

    def __post_init__(self):
        """Ensure that the foreground and background style are converted into a Color or Gradient object"""
        self.fg_style = self._ensure_style(self.fg_style)
        self.bg_style = self._ensure_style(self.bg_style)

    @staticmethod
    def _ensure_style(value: Union[Gradient, Color, str, tuple[int, int, int]]) -> Union[Color, Gradient]:
        """Convert any non Color or Gradient object into a Color assuming an hexadecimal code, RGB tuple or preset string color
        has been specified.

        Args:
            value (Union[Gradient, Color, str, tuple[int, int, int]]): A color value either as a Gradient, Color, rgb tuple, preset or hex code

        Returns:
            Union[Color, Gradient]: Converted style
        """
        if isinstance(value, Color) or isinstance(value, Gradient):
            return value
        return Color(value)

    def apply(self, text: str) -> str:
        """Apply the style properties to the given text

        Args:
            text (str): A string

        Returns:
            str: Stylised text
        """
        bold_ansi = "\033[1m" if self.bold else ""
        darken_ansi = "\033[2m" if self.darken else ""
        italic_ansi = "\033[3m" if self.italic else ""
        underline_ansi = "\033[4m" if self.underline else ""
        double_underline_ansi = "\033[21m" if self.double_underline else ""
        crossed_ainsi = "\033[9m" if self.crossed else ""

        stylised_text = (
            reset_color.to_ansi() +
            bold_ansi +
            italic_ansi +
            underline_ansi +
            crossed_ainsi +
            darken_ansi +
            double_underline_ansi
        )
        fg_chunks, bg_chunks = [], []
        if isinstance(self.fg_style, Gradient):
            fg_chunks = self.fg_style.apply(text)
        elif self.fg_style:
            stylised_text += self.fg_style.to_ansi()
        if isinstance(self.bg_style, Gradient):
            bg_chunks = self.bg_style.apply(text)
        elif self.bg_style:
            stylised_text += self.bg_style.to_ansi(is_background=True)

        fg_idx, bg_idx = 0, 0
        for i, char in enumerate(text):
            if fg_idx < len(fg_chunks) and i == fg_chunks[fg_idx][0]:
                stylised_text += fg_chunks[fg_idx][2].to_ansi()
                fg_idx += 1
            if bg_idx < len(bg_chunks) and i == bg_chunks[bg_idx][0]:
                stylised_text += bg_chunks[bg_idx][2].to_ansi(is_background=True)
                bg_idx += 1
            stylised_text += char

        return stylised_text + reset_color.to_ansi()


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

    return TextStyle(foreground_color, background_color).apply(text)


# Functional Types
DelimiterType = Literal["delimiter"]
KeyType = Literal["key"]
AnyType = Literal["any"]
FunctionalType = Union[Type, DelimiterType, KeyType, AnyType]

def infer_type(self, value: str):
    """Naive type inference that is sufficient for nn.Layer parsing"""
    if value.lower() == "true" or value.lower() == "false":
        return bool
    try:
        int(value)
        return int
    except ValueError: pass
    try:
        float(value)
        return float
    except ValueError: pass
    try:
        if value[0] == "#": return str
    except: pass
    return KeyType

LAYER_SPLITTER = r'\s*[=,]\s*'

@dataclass
class FunctionalStyle:
    """Functional styling class that can split a string with regular expression
    and infer each match to a specific style.
    """
    styles: defaultdict[FunctionalType, TextStyle]
    infer_func = infer_type
    splitter: str

    def __post_init__(self):
        """Ensure that the styles dictionary is a default dict to avoid any inference issue"""
        if not isinstance(self.styles, defaultdict):
            self.styles = defaultdict(TextStyle, self.styles)

    def apply(self, text: str) -> str:
        """Apply the functional style to the text.

        The method uses the splitting regex pattern to separate the text into
        multiple matched strings and their delimiters.
        Then, infer the string to link the corresponding style.

        Args:
            text (str): A string

        Returns:
            str: Stylised string
        """
        matches = re.split(f'({self.splitter})', text)

        result = []
        for match in matches:
            if match == "" or match.isspace():
                continue
            if re.fullmatch(self.splitter, match): result.append((match, False))
            else: result.append((match, True))

        stylised_text = ""
        for res, is_delim in result:
            if not is_delim: stylised_text += self.styles[DelimiterType].apply(res)
            else:
                found_type = self.infer_func(res)
                if found_type not in self.styles:
                    found_type = AnyType
                stylised_text += self.styles[found_type].apply(res)

        return stylised_text