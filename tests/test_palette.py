from torchcolor import Palette, Color, rainbow

class TestPalette:

    def test_rainbow(self):
        print("test")

        palette = Palette([
            Color("#FF0000"),  # Red
            Color("#FF7F00"),  # Orange
            Color("#FFFF00"),  # Yellow
            Color("#00FF00"),  # Green
            Color("#0000FF"),  # Blue
            Color("#A50052"),  # Indigo
            Color("#9400D3")   # Violet
        ])
        print("test")
        print(rainbow("Bonjour je suis un arc-en-ciel!!!!", palette=palette))
        print(rainbow("Bonjour je suis un arc-en-ciel qui se répète!!!!", palette=palette, repeat=True))

if __name__ == "__main__":
    TestPalette().test_rainbow()