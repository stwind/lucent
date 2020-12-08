import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from lucent.nn import MBConv, Swish

# num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio
BLOCK_ARGS = [
    [1, 3, (1, 1), 1, 32, 16, 0.25],
    [2, 3, (2, 2), 6, 16, 24, 0.25],
    [2, 5, (2, 2), 6, 24, 40, 0.25],
    [3, 3, (2, 2), 6, 40, 80, 0.25],
    [3, 5, (1, 1), 6, 80, 112, 0.25],
    [4, 5, (2, 2), 6, 112, 192, 0.25],
    [1, 3, (1, 1), 6, 192, 320, 0.25],
]

# width, depth
SCALES = {
    "b0": (1.0, 1.0),
    "b1": (1.0, 1.1),
    "b2": (1.1, 1.2),
    "b3": (1.2, 1.4),
    "b4": (1.4, 1.8),
    "b5": (1.6, 2.2),
    "b6": (1.8, 2.6),
    "b7": (2.0, 3.1),
    "b8": (2.2, 3.6),
    "l2": (4.3, 5.3),
}

url_map = {
    "b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
    "b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
    "b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
    "b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
    "b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
    "b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
    "b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
    "b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
}


class EfficientNet(nn.Module):
    @staticmethod
    def from_pretrained(name):
        model = EfficientNet()
        sd = model_zoo.load_url(url_map[name])
        model.load_weights(sd)
        return model

    def __init__(
        self,
        cin=3,
        num_classes=1000,
        stem_out=32,
        num_features=1280,
        width=1,
        depth=1,
        divisor=8,
        block_args=BLOCK_ARGS,
        bn_momentum=0.01,
        bn_eps=1e-3,
    ):
        super().__init__()
        ## stem
        cout = self._round_filters(stem_out, width, divisor)
        self._conv_stem = nn.Conv2d(
            cin, cout, kernel_size=3, stride=2, padding=1, bias=False
        )
        self._bn0 = nn.BatchNorm2d(cout, momentum=bn_momentum, eps=bn_eps)

        ## blocks
        self._blocks = nn.ModuleList([])
        for args in block_args:
            self._blocks.append(
                MBConv(
                    cin=self._round_filters(args[4], width, divisor),
                    cout=self._round_filters(args[5], width, divisor),
                    ks=args[1],
                    stride=args[2],
                    expand_ratio=args[3],
                    se_ratio=args[6],
                )
            )
            for i in range(math.ceil(args[0] - 1 * depth)):
                self._blocks.append(
                    MBConv(
                        cin=self._round_filters(args[5], width, divisor),
                        cout=self._round_filters(args[5], width, divisor),
                        ks=args[1],
                        stride=(1, 1),
                        expand_ratio=args[3],
                        se_ratio=args[6],
                    )
                )

        ## head
        cout = self._round_filters(num_features, width, divisor)
        self._conv_head = nn.Conv2d(block_args[-1][5], cout, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(cout, momentum=bn_momentum, eps=bn_eps)

        # classifier
        self.classifier = nn.Linear(cout, num_classes)
        self._swish = Swish()

    def features(self, x):
        x = self._swish(self._bn0(self._conv_stem(x)))
        for block in self._blocks:
            x = block(x)
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        x = self.features(inputs)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def _round_filters(self, filters, width, divisor):
        filters *= width
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
            new_filters += divisor
        return int(new_filters)

    def load_weights(self, source):
        """
        load models from https://github.com/lukemelas/EfficientNet-PyTorch
        """
        mod_keys = {
            "conv": ["weight"],
            "bn": [
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ],
            "fc": ["weight", "bias"],
        }

        def state_dict(mod, name):
            return {x: source["{}.{}".format(name, x)] for x in mod_keys[mod]}

        def block_key(i, name):
            return "_blocks.{}.{}".format(i, name)

        ## stem
        self._conv_stem.load_state_dict(state_dict("conv", "_conv_stem"))
        self._bn0.load_state_dict(state_dict("bn", "_bn0"))

        ## blocks
        for i, block in enumerate(self._blocks):
            if block._expand_conv:
                block._expand_conv.load_state_dict(
                    state_dict("conv", block_key(i, "_expand_conv"))
                )
                block._bn0.load_state_dict(state_dict("bn", block_key(i, "_bn0")))
            block._depthwise_conv.load_state_dict(
                state_dict("conv", block_key(i, "_depthwise_conv"))
            )
            block._bn1.load_state_dict(state_dict("bn", block_key(i, "_bn1")))
            block._se.load_state_dict(
                {
                    a: source[block_key(i, b)]
                    for a, b in [
                        ("_reduce.weight", "_se_reduce.weight"),
                        ("_reduce.bias", "_se_reduce.bias"),
                        ("_expand.weight", "_se_expand.weight"),
                        ("_expand.bias", "_se_expand.bias"),
                    ]
                }
            )
            block._project_conv.load_state_dict(
                state_dict("conv", block_key(i, "_project_conv"))
            )
            block._bn2.load_state_dict(state_dict("bn", block_key(i, "_bn2")))

        ## head
        self._conv_head.load_state_dict(state_dict("conv", "_conv_head"))
        self._bn1.load_state_dict(state_dict("bn", "_bn1"))

        ## classifier
        self.classifier.load_state_dict(state_dict("fc", "_fc"))