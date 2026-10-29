from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FRAME_DIR = Path(r"C:\Users\haruh\Documents\Codex\フレーム")
OUTPUT = Path(__file__).resolve().parent / "gx010202_flow_scan_contact_sheet.jpg"
FRAMES = [289, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950]


def main() -> None:
    thumb_size = (676, 380)
    label_height = 38
    sheet = Image.new("RGB", (thumb_size[0] * 3, (thumb_size[1] + label_height) * 4), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 25) if font_path.exists() else ImageFont.load_default()
    for index, frame in enumerate(FRAMES):
        path = FRAME_DIR / f"GX010202_timestamp_JST{frame:04d}.jpg"
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        col, row = index % 3, index // 3
        x, y = col * thumb_size[0], row * (thumb_size[1] + label_height)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y + thumb_size[1], x + thumb_size[0], y + thumb_size[1] + label_height), fill="white")
        draw.text((x + 10, y + thumb_size[1] + 3), f"frame {frame}", fill="black", font=font)
    sheet.save(OUTPUT, quality=92)
    print(OUTPUT)


if __name__ == "__main__":
    main()
