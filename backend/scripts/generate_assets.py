import os
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = os.path.join(os.getcwd(), "assets")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_image(name: str, title: str):
    ensure_dir(ASSETS_DIR)
    img = Image.new("RGB", (1200, 800), (10, 14, 26))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 64)
        font_sub = ImageFont.truetype("arial.ttf", 32)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    draw.text((60, 60), title, fill=(199, 210, 254), font=font_title)
    draw.text((60, 150), "Annotated prediction preview", fill=(163, 163, 179), font=font_sub)
    out_path = os.path.join(ASSETS_DIR, name)
    img.save(out_path, "PNG")
    return out_path

def main():
    items = [
        ("overview.png", "Action Recognition by Abdullah"),
        ("cutting_vegetables.png", "cutting_vegetables"),
        ("smoking.png", "smoking"),
        ("writing_on_a_book.png", "writing_on_a_book"),
        ("running.png", "running"),
    ]
    paths = [make_image(n, t) for n, t in items]
    print("\n".join(paths))

if __name__ == "__main__":
    main()
