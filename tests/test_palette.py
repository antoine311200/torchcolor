from torchcolor.palette import Palette
from torchcolor import Color
from torchcolor.style import TextStyle
from torchcolor.gradient import Gradient

class TestPalette:

    def test_rainbow(self):

        palette = Palette("niji", [
            Color("#FF0000"),  # Red
            Color("#FF7F00"),  # Orange
            Color("#FFFF00"),  # Yellow
            Color("#00FF00"),  # Green
            Color("#0000FF"),  # Blue
            Color("#A50052"),  # Indigo
            Color("#9400D3")   # Violet
        ], disable_registry=True)

        print(TextStyle(Gradient(palette, interpolate=False)).apply("Bonjour je suis un arc-en-ciel!!!!"))
        print(TextStyle(Gradient(Palette.get_palette("retro_neon"), interpolate=False, repeat=True)).apply("Bonjour je suis un arc-en-ciel qui se répète!!!!"))

        text = "Je suis un magnifique texte d'exemple avec un dégradé stylax!"

        for palette in Palette._registry.values():
            print(TextStyle(Gradient(palette, interpolate=False, repeat=True, window_size=5)).apply(text))

        print(TextStyle(Gradient(Palette.get_palette("rainbow"), interpolate=False)).apply(text))

        print()
        print("########################################")
        print("############### GRADIENT ###############")
        print("########################################")
        print()
        for palette in Palette._registry.values():
            print(TextStyle(Gradient(palette, interpolate=True)).apply(text))

        long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam consequat lectus eu quam iaculis, vel blandit ligula sagittis. Nam et tellus vel risus fringilla auctor ut vitae ligula. In vitae rutrum erat. Donec vel dolor faucibus ex mattis convallis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse in varius orci. Vestibulum ut lacus dignissim, tincidunt ex ut, fermentum ligula. Vivamus tempor metus magna. Maecenas faucibus dignissim tincidunt. Integer luctus sollicitudin eros non mollis."
        print()
        for palette in Palette._registry.values():
            print(TextStyle(Gradient(palette, interpolate=True)).apply(long_text))


        print()
        print("###################################################################")
        print("############### Foreground and background Gradients ###############")
        print("###################################################################")
        print()
        
        for fg_palette in Palette._registry.values():
            for bg_palette in Palette._registry.values():
                if fg_palette != bg_palette:
                    print(TextStyle(Gradient(fg_palette, interpolate=True), Gradient(bg_palette, interpolate=True)).apply(text))


if __name__ == "__main__":
    TestPalette().test_rainbow()