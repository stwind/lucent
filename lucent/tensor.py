import torch
import torch.nn.functional as F
from contextlib import AbstractContextManager
from lucent.utils import binomial_filter


class Hooks(AbstractContextManager):
    def __init__(self, tensor, funcs):
        self.hooks = []
        for f in funcs:
            self.hooks.append(tensor.register_hook(f))

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()


def lap_normalizer(lap_n, filter_size=5):
    kernel = torch.from_numpy(binomial_filter(filter_size)).permute(2, 3, 0, 1)
    return lambda g: lap_normalize(g, kernel.to(g.device), lap_n)


def std_normalizer():
    return lambda g: g / g.std()


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


def img_from_numpy(arr2d, unsqueeze=True):
    img = to_chw(torch.from_numpy(arr2d))
    if img.dim() == 3 and unsqueeze:
        img = img.unsqueeze(0)
    return img


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

    return F.conv_transpose2d(
        F.pad(x, [1, 1 - opw, 1, 1 - oph]),
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