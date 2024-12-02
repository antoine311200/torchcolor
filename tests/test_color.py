import pytest

from torchcolor import print_color
from torchcolor.print import print_more
from torchcolor.palette import Palette
from torchcolor.gradient import Gradient
from torchcolor.style import TextStyle, colorize

class TestColor:

    def test_print_color(self):
        print_color("Hello World!", text_color="red", bg_color="white")
        print(colorize("Hello World!", text_color="red", bg_color="white"))
        print_color("This test is #941D1D with background (234, 193, 71)", text_color="#941D1D", bg_color=(234, 193, 71))

        assert colorize("Hello World!", text_color="red", bg_color="white") == "\033[0m\033[31m\033[47mHello World!\033[0m"

    def test_text_color(self):
        palette = Palette.get_palette("warm_sunset")
        stylised_color = TextStyle(
            fg_style=palette[4], bg_style=(24, 201, 120),
            bold=True, underline=True
        )
        print(stylised_color.apply("Salut à tous les zamis!"))
        stylised_color = TextStyle(
            fg_style=palette[4], bg_style=(24, 201, 120),
            bold=False, underline=False
        )
        print(stylised_color.apply("Salut à tous les zamis!"))

        print(TextStyle(Gradient("warm_sunset"), Gradient("ocean_breeze")).apply("Bonjour la compagnie ! Je suis heureux de vous parler aujourd'hui"))

        print()
        print_more(
            "Salut à tous", TextStyle("red", "blue"),
            "je suis super impatient de", TextStyle("#f2d4aa"),
            "rentrer",
            "en", TextStyle("yellow"),
            "FRANCE", TextStyle(Gradient("warm_sunset"), crossed=True, bold=True, underline=True, italic=True),
            "c'est vraiment un truc de",
            "fou furieux", TextStyle((255, 0, 0), double_underline=True),
            "de",
            "fou furieux", TextStyle((255, 0, 0), darken=True),
        )
        from torchcolor.color import Color
        print()
        little_style = TextStyle(fg_style=Gradient(Palette.get("rainbow")), bg_style=Color((75, 75, 75)), crossed=True)

        print_more(
            "Lorem ipsum ", TextStyle("red"),
            "dolor sit amet", TextStyle(Gradient(Palette.get("warm_sunset")), underline=True),
            ", consectetur adipiscing ",
            "elit", TextStyle(bg_style="bright cyan", italic=True),
            ". ",
            "Nullam consequat", little_style,
            " lectus ",
            "eu quam iaculis", little_style,
            ", ",
            "vel blandit ligula sagittis.", TextStyle("green", darken=True),
            sep=""
        )


        print()
        color_red = Color("red")
        color_rgb = Color((120, 240, 70))
        # color_hex = Color("#F3AA5B")
        color_hex = Color("#F3115B")


        style1 = TextStyle(fg_style=color_red, bg_style=color_rgb, italic=True, double_underline=True)
        style2 = TextStyle(fg_style=color_hex, bg_style="bright cyan", crossed=True)

        text = "This text has a torchcolor styling!"
        print(style1.apply(text))
        print(style2.apply(text))
        print()

if __name__ == "__main__":
    TestColor().test_print_color()
    TestColor().test_text_color()