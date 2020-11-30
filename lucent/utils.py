import io
import os
import hashlib
import tempfile
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontman
import torch
from torchvision.utils import make_grid
from PIL import Image, ImageFont, ImageDraw


def print_topk(out, labels, k=5):
    probs = out.softmax(dim=1)[0]
    for i in out.topk(k).indices[0].tolist():
        print("{:<75} ({:.2f}%)".format(labels[i], probs[i] * 100))


def plot_img(img, figsize=(3, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_axis_off()
    with plt.rc_context({"savefig.pad_inches": 0}):
        plt.show()


def plot_img_batch(imgs):
    plot_img(to_hwc(make_grid(imgs)))


def to_hwc(x):
    ndim = x.dim()
    assert ndim in (3, 4)
    if x.dim() == 4:
        return x.permute(0, 2, 3, 1)
    return x.permute(1, 2, 0)


def to_chw(x):
    ndim = x.dim()
    assert ndim in (3, 4)
    if x.dim() == 4:
        return x.permute(0, 3, 1, 2)
    return x.permute(2, 0, 1)


def to_var(x, device="cpu"):
    return x.to(device).requires_grad_(True)


def minmax_scale(x):
    mn, mx = x.max(), x.min()
    return (x - mn) / (mx - mn)


def decorrelate(img, corr):
    return torch.einsum("nchw,cd->ndhw", img, torch.from_numpy(corr)).sigmoid()


def center_crop(img, size):
    return img[size:-size, size:-size, :]


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


def stitch_images(images, nrow=1, gap=1, bg="black"):
    n = len(images)
    ncol = n // nrow
    w, h = images[0].width, images[0].height
    dst = Image.new(
        "RGB", (w * nrow + gap * (nrow + 1), w * ncol + gap * (ncol + 1)), bg
    )
    for i, src in enumerate(images):
        ix, iy = divmod(i, ncol)
        x = gap + ix * (h + gap)
        y = gap + iy * (w + gap)
        dst.paste(src, (x, y))

    return dst


def fetch(url, fp=None):
    if not fp:
        fp = os.path.join(
            tempfile.gettempdir(), hashlib.md5(url.encode("utf-8")).hexdigest()
        )
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print("fetching {}".format(url))
        dat = requests.get(url).content
        with open(fp + ".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp + ".tmp", fp)
    return dat, fp


def fetch_image(url):
    resp = requests.get(
        "https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/examples/simple/img2.jpg"
    )
    return PIL.Image.open(io.BytesIO(resp.content))