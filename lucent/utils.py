import io
import os
import hashlib
import tempfile
import requests
import urllib.request
import tarfile
import numpy as np
import matplotlib.pyplot as plt
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


def to_var(x, device="cpu"):
    return x.to(device).requires_grad_(True)


def minmax_scale(x, axis=None):
    mx = x.max(axis=axis, keepdims=True)
    mn = x.min(axis=axis, keepdims=True)
    return (x - mn) / (mx - mn)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def decorrelate(img, corr):
    return sigmoid(img.dot(corr))


def center_crop(img, size):
    return img[size:-size, size:-size, :]


def fetch_url(url, fp=None):
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


def binomcoeffs(n):
    return (np.poly1d([1, 1]) ** n).coeffs


def binomial_filter(n, channel=3, dtype=np.float32):
    coef = binomcoeffs(n - 1)
    k = np.outer(coef, coef)
    return np.expand_dims(k / k.sum(), (2, 3)).astype(dtype) * np.eye(
        channel, dtype=dtype
    )


def download_tar(url, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    stream = urllib.request.urlopen(url)
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        tar.extractall(path=path)