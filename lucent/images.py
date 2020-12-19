import io
import os
import requests
import numpy as np
import matplotlib.font_manager as fontman
from PIL import Image, ImageFont, ImageDraw

from lucent.imagenet import COLOR_CORRELATION
from lucent.utils import decorrelate


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


def make_normal(
    size, mean=0, std=0.01, batch=1, dtype=np.float32, corr=COLOR_CORRELATION
):
    h, w = (size, size) if isinstance(size, int) else size
    img = np.random.normal(mean, std, (batch, h, w, 3)).astype(dtype)
    if corr is not None:
        img = decorrelate(img, corr)
    return img


def rfft2d_freqs(h, w):
    fy = np.fft.fftfreq(h)[:, np.newaxis]
    fx = np.fft.fftfreq(w)[: w // 2 + 1 + (w % 2)]
    return np.sqrt(fx * fx + fy * fy)


def make_fft(
    size,
    sd=0.01,
    batch=1,
    channel=3,
    decay_power=1,
    dtype=np.float32,
    corr=COLOR_CORRELATION,
):
    h, w = (size, size) if isinstance(size, int) else size
    freqs = rfft2d_freqs(h, w)

    init_size = (2, batch, channel) + freqs.shape
    init_val = np.random.normal(size=init_size, scale=sd).astype(dtype)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w * h)

    spectrum = init_val[0] + 1j * init_val[1]
    spectrum *= scale

    img = np.fft.irfft2(spectrum).astype(dtype).transpose((0, 2, 3, 1))
    img = img[:batch, :h, :w, :channel]
    img /= 4.0

    if corr is not None:
        img = decorrelate(img, corr)
    return img