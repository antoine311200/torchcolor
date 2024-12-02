import pytest

from torchcolor.style import TextStyle, FunctionalStyle, DelimiterType, KeyType

class TestFunctional:

    def test_functional(self):
        style = FunctionalStyle(
            splitter=r'\s*[=,]\s*',
            styles={
                KeyType: TextStyle("red", underline=True),
                DelimiterType: TextStyle(double_underline=True),
                bool: TextStyle("blue", "bright red")
            })
        print(style.lex("in_features=10, out_features=10, bias=True"))

if __name__ == "__main__":
    TestFunctional().test_functional()