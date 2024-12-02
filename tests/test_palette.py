from torchcolor.palette import Palette
from torchcolor import Color
from torchcolor.style import TextStyle
from torchcolor.gradient import Gradient

from copy import deepcopy

class TestPalette:

    def test_rainbow(self):

        palette = Palette("niji", deepcopy(Palette.get_palette("rainbow").colors))
        gradient = Gradient(palette)
        style = TextStyle(gradient, Color((70, 70, 70)))

        print(style.apply("This is a palette test"))

        palette.colors = [Color((102, 45, 210)), Color((65, 84, 179)), Color((180, 240, 120))]
        print(style.apply("This is a palette test"))

        gradient.reverse = True
        print(style.apply("This is a palette test"))

        gradient.interpolate = False
        gradient.repeat = True
        style.bg_style = None
        print(style.apply("This is a palette test"))


if __name__ == "__main__":
    TestPalette().test_rainbow()