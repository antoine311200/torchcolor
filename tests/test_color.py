import pytest

from torchcolor import print_color, colorize

class TestColor:

    def test_print_color(self):
        print_color("Hello World!", text_color="red", bg_color="white")
        print(colorize("Hello World!", text_color="red", bg_color="white"))

        assert colorize("Hello World!", text_color="red", bg_color="white") == "\033[31m\033[47mHello World!\033[0m"