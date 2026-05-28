"""Generate the BMP banner images Inno Setup needs.

Inno Setup expects:
  WizardImageFile     — 164×314 BMP shown on Welcome / Finish pages
  WizardSmallImageFile — 55×58 BMP shown in the corner of inner pages

We render both in the same arc-reactor aesthetic as the icon.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent


def _draw_arc_reactor(img: Image.Image, cx: int, cy: int, radius: int,
                      ring_color: tuple[int, int, int],
                      glow_color: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 fill=(20, 20, 38, 255), outline=ring_color, width=4)
    inner = int(radius * 0.65)
    draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner],
                 outline=glow_color, width=2)
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4],
                 fill=ring_color)


def make_welcome() -> None:
    """164x314 vertical banner with full arc-reactor + tagline."""
    w, h = 164, 314
    img = Image.new("RGB", (w, h), (22, 22, 42))
    draw = ImageDraw.Draw(img)

    # Subtle gradient via vertical stripes
    for y in range(h):
        t = y / h
        r = int(22 + 8 * (1 - t))
        g = int(22 + 6 * (1 - t))
        b = int(42 + 18 * (1 - t))
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    _draw_arc_reactor(img, w // 2, 100, 50,
                      ring_color=(79, 195, 247),
                      glow_color=(79, 195, 247))

    try:
        font_big = ImageFont.truetype("segoeuib.ttf", 22)
    except OSError:
        font_big = ImageFont.load_default()
    try:
        font_small = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        font_small = ImageFont.load_default()

    # JARVIS title
    title = "JARVIS"
    bbox = draw.textbbox((0, 0), title, font=font_big)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, 175), title, font=font_big, fill=(255, 255, 255))

    # Tagline
    tagline = "Your private\nAI assistant"
    for i, line in enumerate(tagline.split("\n")):
        bbox = draw.textbbox((0, 0), line, font=font_small)
        lw = bbox[2] - bbox[0]
        draw.text(((w - lw) // 2, 210 + i * 16), line, font=font_small,
                  fill=(170, 180, 200))

    img.save(OUT_DIR / "welcome.bmp", "BMP")
    print(f"Saved {OUT_DIR / 'welcome.bmp'}")


def make_small() -> None:
    """55x58 small image shown on inner wizard pages."""
    w, h = 55, 58
    img = Image.new("RGB", (w, h), (22, 22, 42))
    _draw_arc_reactor(img, w // 2, h // 2, 22,
                      ring_color=(79, 195, 247),
                      glow_color=(79, 195, 247))
    img.save(OUT_DIR / "header.bmp", "BMP")
    print(f"Saved {OUT_DIR / 'header.bmp'}")


if __name__ == "__main__":
    make_welcome()
    make_small()
