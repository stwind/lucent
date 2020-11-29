import io
import os
import hashlib
import tempfile
import requests
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
from torchvision.utils import make_grid


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


def stitch_images(images, nrow=1, gap=1, bg=0):
    n = len(images)
    h, w, c = images[0].shape
    ncol = n // nrow
    img = np.zeros(
        (h * nrow + gap * (nrow + 1), w * ncol + gap * (ncol + 1), c), dtype=np.float32
    )
    img.fill(bg)
    for i, d in enumerate(images):
        ix, iy = divmod(i, ncol)
        x = gap + ix * (h + gap)
        y = gap + iy * (w + gap)
        img[x : x + h, y : y + w, :] = d

    return img


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