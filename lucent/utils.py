import io
import os
import hashlib
import tempfile
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontman
import torch
import fastprogress.fastprogress
import torch.nn.functional as F
from torchvision.utils import make_grid
from PIL import Image, ImageFont, ImageDraw
from IPython import display


def progress_bar(**kwargs):
    opts = {"leave": False}
    opts.update(**kwargs)
    return lambda gen: fastprogress.progress_bar(gen, **opts)


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
        "RGB", (w * ncol + gap * (ncol + 1), h * nrow + gap * (nrow + 1)), bg
    )
    for i, src in enumerate(images):
        ix, iy = divmod(i, ncol)
        y = gap + ix * (h + gap)
        x = gap + iy * (w + gap)
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
    resp = requests.get(url)
    return Image.open(io.BytesIO(resp.content))


def display_pil_image(img):
    bio = io.BytesIO()
    img.save(bio, format="png")
    display.display(display.Image(bio.getvalue(), format="png", retina=True))


def binomcoeffs(n):
    return (np.poly1d([1, 1]) ** n).coeffs


def binomial_filter(n, channel=3):
    coef = binomcoeffs(n - 1)
    k = np.outer(coef, coef)
    return np.expand_dims(k / k.sum(), (2, 3)).astype(np.float32) * np.eye(
        channel, dtype=np.float32
    )


def padding_same(in_size, kernel_size, stride=1, dilation=1):
    filter_size = (kernel_size - 1) * dilation + 1
    out_size = (in_size + stride - 1) // stride
    return max(0, (out_size - 1) * stride + filter_size - in_size)


def conv_transpose2d_to(
    x, weight, output_shape, bias=None, stride=1, dilation=1, groups=1
):
    """
    same behavior as `tf.nn.conv2d_transpose`
    """
    oh, ow = output_shape[-2:]
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    kh, kw = (kh - 1) * dilation + 1, (kw - 1) * dilation + 1
    sh, sw = (stride, stride) if isinstance(stride, int) else stride

    offset_h = (ih - 1) * sh + kh - oh
    offset_w = (iw - 1) * sw + kw - ow
    oph, opw = offset_h % 2, offset_w % 2

    x = F.pad(x, [1, 1 - opw, 1, 1 - oph])
    return F.conv_transpose2d(
        x,
        weight,
        bias=bias,
        stride=(sh, sw),
        padding=(offset_h, offset_w),
        output_padding=(oph, opw),
        dilation=dilation,
        groups=groups,
    )


def lap_split(img, kernel, scale=4):
    n, c, h, w = img.size()

    kh, kw = kernel.size()[-2:]
    ph, pw = padding_same(h, kh, 2), padding_same(w, kw, 2)

    x = F.pad(img, [0, pw % 2, 0, ph % 2]) if (ph % 2 or pw % 2) else img
    lo = F.conv2d(x, kernel, stride=2, padding=(ph // 2, pw // 2))
    lo2 = conv_transpose2d_to(lo, kernel * scale, (n, c, h, w), stride=2)
    return lo, img - lo2


def lap_split_n(img, kernel, n=4):
    levels = []
    for i in range(n):
        img, hi = lap_split(img, kernel)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]


def lap_merge(levels, kernel, scale=4):
    img = levels[0]
    for hi in levels[1:]:
        img = conv_transpose2d_to(img, kernel * scale, hi.size(), stride=2) + hi
    return img


def normalize_std(img, eps=torch.tensor(1e-10)):
    std = img.square().mean().sqrt()
    return img / std.maximum(eps.to(std.device))


def lap_normalize(img, kernel, n=4):
    tlevels = lap_split_n(img, kernel, n)
    tlevels = list(map(normalize_std, tlevels))
    return lap_merge(tlevels, kernel)