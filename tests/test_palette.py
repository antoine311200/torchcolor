from torchcolor.palette import Palette, rainbow
from torchcolor import Color

class TestPalette:

    def test_rainbow(self):
        print("test")

        palette = Palette("niji", [
            Color("#FF0000"),  # Red
            Color("#FF7F00"),  # Orange
            Color("#FFFF00"),  # Yellow
            Color("#00FF00"),  # Green
            Color("#0000FF"),  # Blue
            Color("#A50052"),  # Indigo
            Color("#9400D3")   # Violet
        ], disable_registry=True)
        print("test")
        print(rainbow("Bonjour je suis un arc-en-ciel!!!!", palette=palette))
        print(rainbow("Bonjour je suis un arc-en-ciel qui se répète!!!!", palette=Palette.get_palette("retro_neon"), repeat=True))

        text = "Je suis un magnifique texte d'exemple avec un dégradé stylax!"

        for palette in Palette._registry.values():
            print(rainbow(text, palette, repeat=True, window_size=5))

        print(rainbow(text, palette=Palette.get_palette("rainbow"), gradient=False))

        print()
        print("########################################")
        print("############### GRADIENT ###############")
        print("########################################")
        print()
        for palette in Palette._registry.values():
            print(rainbow(text, palette, gradient=True))

        long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam consequat lectus eu quam iaculis, vel blandit ligula sagittis. Nam et tellus vel risus fringilla auctor ut vitae ligula. In vitae rutrum erat. Donec vel dolor faucibus ex mattis convallis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse in varius orci. Vestibulum ut lacus dignissim, tincidunt ex ut, fermentum ligula. Vivamus tempor metus magna. Maecenas faucibus dignissim tincidunt. Integer luctus sollicitudin eros non mollis."
        print()
        for palette in Palette._registry.values():
            print(rainbow(long_text, palette, gradient=True))

if __name__ == "__main__":
    TestPalette().test_rainbow()