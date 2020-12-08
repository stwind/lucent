import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from lucent.utils import fetch


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
IMAGE_SIZE = 224

## https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/color.py#L24
color_correlation_svd_sqrt = np.array(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]], dtype=np.float32
)
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
COLOR_CORRELATION = color_correlation_normalized.T

normalize = Normalize(MEAN, STD)
denormalize = Normalize(-MEAN / STD, 1 / STD)

preprocess = Compose([Resize(IMAGE_SIZE), ToTensor(), normalize])


def deprocess(img):
    return denormalize(img.detach().cpu()).clip(0, 1)


LABEL_URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"


def load_labels():
    lbls_dat, _ = fetch(LABEL_URL)
    return eval(lbls_dat)