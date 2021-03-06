import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseConv2d(nn.Conv2d):
    def __init__(self, cin, ks=3, **kwargs):
        super().__init__(cin, cin, ks, padding=(ks - 1) // 2, groups=cin, **kwargs)


class PointWiseConv2d(nn.Conv2d):
    def __init__(self, cin, cout, **kwargs):
        super().__init__(cin, cout, 1, stride=1, bias=False, **kwargs)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcite(nn.Module):
    def __init__(self, cin, cout, act=nn.ReLU()):
        super().__init__()
        self._reduce = nn.Conv2d(cin, cout, 1)
        self._expand = nn.Conv2d(cout, cin, 1)
        self._act = act

    def forward(self, inputs):
        x = self._act(self._reduce(F.adaptive_avg_pool2d(inputs, 1)))
        x = torch.sigmoid(self._expand(x))
        return x * inputs


class MBConv(nn.Module):
    def __init__(
        self,
        cin,
        cout,
        ks=3,
        stride=1,
        expand_ratio=4,
        se_ratio=0.25,
        bn_momentum=0.01,
        bn_eps=1e-3,
    ):
        super().__init__()
        eout = round(cin * expand_ratio)
        if expand_ratio != 1:
            self._expand_conv = PointWiseConv2d(cin, eout)
            self._bn0 = nn.BatchNorm2d(eout, momentum=bn_momentum, eps=bn_eps)
        else:
            self._expand_conv = None

        self._swish = Swish()

        self._depthwise_conv = DepthWiseConv2d(eout, ks, stride=stride, bias=False)
        self._bn1 = nn.BatchNorm2d(eout, momentum=bn_momentum, eps=bn_eps)

        sout = max(1, int(cin * se_ratio))
        self._se = SqueezeExcite(eout, sout, act=self._swish)

        self._project_conv = PointWiseConv2d(eout, cout)
        self._bn2 = nn.BatchNorm2d(cout, momentum=bn_momentum, eps=bn_eps)

    def forward(self, inputs):
        x = inputs
        if self._expand_conv:
            x = self._swish(self._bn0(self._expand_conv(x)))

        x = self._swish(self._bn1(self._depthwise_conv(x)))
        x = self._se(x)
        x = self._bn2(self._project_conv(x))

        if x.shape == inputs.shape:
            x += inputs

        return x


class RandomCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        h, w = img.size(2) - self.size, img.size(3) - self.size
        oh, ow = np.random.randint(0, self.size // 2 + 1, 2)
        return img[:, :, oh : oh + h, ow : ow + w]


class Jitter(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        sy, sx = np.random.randint(-self.size, self.size, 2)
        return input.roll(shifts=(sy, sx), dims=(-2, -1))


class Pad(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        w = self.size
        return F.pad(img, (w, w, w, w), "constant", 0.5)


class RandomScale(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.scales = scales

    def forward(self, img):
        scale = np.random.choice(self.scales)
        return F.interpolate(
            img,
            mode="bilinear",
            align_corners=True,
            scale_factor=(scale, scale),
            recompute_scale_factor=True,
        )


class RandomRotate(nn.Module):
    def __init__(self, rads):
        super().__init__()
        self.rads = rads

    def _rot_mat(self, theta):
        theta = torch.tensor(theta)
        return torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
            ]
        )

    def forward(self, img):
        rad = np.random.choice(self.rads)
        size = img.size()
        mat = self._rot_mat(rad).unsqueeze(0).expand((size[0], 2, 3))
        grid = F.affine_grid(mat, size, align_corners=False).to(img.device)
        return F.grid_sample(img, grid, align_corners=False)


class TotalVariation(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, img):
        w_v = torch.sum(torch.pow(img[..., :, :-1] - img[..., :, 1:], self.beta / 2.0))
        h_v = torch.sum(torch.pow(img[..., :-1, :] - img[..., 1:, :], self.beta / 2.0))
        return h_v + w_v