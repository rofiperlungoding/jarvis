"""Generate a minimal JARVIS icon (256x256 ICO).

Design: dark circle background (#1a1a2e) with a glowing blue arc reactor
circle (#4fc3f7) and a centered "J" letterform in white. Clean, minimal,
enterprise-looking.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 256
OUT = Path(__file__).parent / "jarvis.ico"


def generate() -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle
    margin = 8
    draw.ellipse(
        [margin, margin, SIZE - margin, SIZE - margin],
        fill=(26, 26, 46, 255),
    )

    # Glowing arc reactor ring
    ring_margin = 28
    draw.ellipse(
        [ring_margin, ring_margin, SIZE - ring_margin, SIZE - ring_margin],
        outline=(79, 195, 247, 255),
        width=6,
    )

    # Inner glow ring (thinner, lighter)
    inner_margin = 48
    draw.ellipse(
        [inner_margin, inner_margin, SIZE - inner_margin, SIZE - inner_margin],
        outline=(79, 195, 247, 120),
        width=2,
    )

    # Center "J" letter
    try:
        font = ImageFont.truetype("segoeui.ttf", 90)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 90)
        except OSError:
            font = ImageFont.load_default()

    text = "J"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (SIZE - tw) // 2 - bbox[0]
    ty = (SIZE - th) // 2 - bbox[1] - 4
    draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)

    # Small dot in center (arc reactor core)
    cx, cy = SIZE // 2, SIZE // 2
    draw.ellipse([cx - 4, cy + 38, cx + 4, cy + 46], fill=(79, 195, 247, 200))

    # Save as ICO with multiple sizes
    sizes = [img.resize((s, s), Image.LANCZOS) for s in (16, 32, 48, 64, 128, 256)]
    sizes[-1].save(str(OUT), format="ICO", append_images=sizes[:-1])
    print(f"Icon saved: {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    generate()
