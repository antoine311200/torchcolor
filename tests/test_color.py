import pytest

from torchcolor import print_color, colorize

class TestColor:

    def test_print_color(self):
        print_color("Hello World!", text_color="red", bg_color="white")
        print(colorize("Hello World!", text_color="red", bg_color="white"))
        print_color("This test is #941D1D with background (234, 193, 71)", text_color="#941D1D", bg_color=(234, 193, 71))

        assert colorize("Hello World!", text_color="red", bg_color="white") == "\033[0m\033[31m\033[47mHello World!\033[0m"


if __name__ == "__main__":
    TestColor().test_print_color()