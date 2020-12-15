import io
import os
import requests

import matplotlib.font_manager as fontman
from PIL import Image, ImageFont, ImageDraw

from lucent.utils import fetch_url


def get_font_files(name):
    return list(
        filter(
            lambda path: name.lower() in os.path.basename(path).lower(),
            fontman.findSystemFonts(),
        )
    )


def draw_text(
    img,
    text,
    position=(8, 5),
    fg="white",
    bg="black",
    font_name="LiberationSansNarrow-Bold.ttf",
    font_size=13,
    pad=2,
):
    font_files = get_font_files(font_name)
    if not font_files:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_files[0], font_size)
    draw = ImageDraw.Draw(img)
    x, y = position
    w, h = font.getsize(text)
    draw.rectangle((x - pad, y, x + w + pad, y + h + pad), fill=bg)
    draw.text(position, text, fill=fg, font=font)
    return img


def stitch(images, nrow=1, gap=1, bg="black"):
    n = len(images)
    ncol = n // nrow
    w, h = images[0].width, images[0].height
    dst = Image.new(
        "RGB", (w * ncol + gap * (ncol + 1), h * nrow + gap * (nrow + 1)), bg
    )
    for i, src in enumerate(images):
        ix, iy = divmod(i, ncol)
        y = gap + ix * (h + gap)
        x = gap + iy * (w + gap)
        dst.paste(src, (x, y))

    return dst


def fetch(url):
    resp = requests.get(url)
    return Image.open(io.BytesIO(resp.content))
