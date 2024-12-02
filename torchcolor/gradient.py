from dataclasses import dataclass
from typing import List, Type, Union

from .palette import Palette
from .color import Color, reset_color

GradientChunk = Type[tuple[int, int, Color]]

@dataclass
class Gradient:
    """Dataclass handling Gradient style from a given color palette and different display properties."""
    palette: Union[Palette, str]
    reverse: bool = False
    interpolate: bool = True
    repeat: bool = False
    window_size: int = 1

    def __post_init__(self):
        self.palette = self._ensure_palette(self.palette)

    def _ensure_palette(self, palette: Union[Palette, str]) -> Palette:
        """Ensure that the palette property is of type Palette otherwise get the palette by its name if it exists.

        Args:
            palette (Union[Palette, str]): A color palette or the name of a registered palette

        Returns:
            Palette: A color palette
        """
        if isinstance(palette, Palette): return palette
        return Palette.get_palette(palette)

    def apply(self, text) -> List[GradientChunk]:
        """Apply the gradient effect to a text using a palette of colors.

        Args:
            palette (Palette): Palette of colors to use.
            text (str): Text to colorize.
            repeat (bool): Whether to repeat colors from the palette. Defaults to True.

        Returns:
            List[GradientChunk]: A list of gradient chunk with start and end index with the proper color to apply.
        """
        chunks = []
        n_colors = len(self.palette.colors)
        length = len(text)

        if self.interpolate:
            colors = self.palette.generate_gradient(n=length)
        else:
            colors = self.palette.colors
        if self.reverse: colors = colors[::-1]

        if self.repeat:
            # Cycle through colors for each character
            for i in range(length):
                color = colors[i//self.window_size % len(colors)]  # Cycle through the palette
                chunks.append((i, i+1, color))
        else:
            # Divide text into contiguous segments based on the palette
            segment_size = max(1, length // n_colors) if not self.interpolate else 1
            for i, color in enumerate(colors):
                start = i * segment_size
                end = start + segment_size if i < n_colors - 1 or self.interpolate else length
                chunks.append((start, end, color))

        return chunks
