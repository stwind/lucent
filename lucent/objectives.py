import torch
import torch.nn as nn

from lucent.viz import T


class Objective(nn.Module):
    def __init__(self, modules, regularizers=[], device="cpu"):
        super().__init__()
        self.t = T(modules)
        self.regularizers = regularizers
        self.device = device

    def regularize(self, inputs):
        res = torch.tensor(0.0).to(self.device)
        for reg, w in self.regularizers:
            res += w * reg(inputs)
        return res


class Channel(Objective):
    def __init__(self, module, index, weight=-1, **kwargs):
        super().__init__([module], **kwargs)
        self.index = index
        self.weight = weight

    def forward(self, inputs):
        loss = self.t(0)[:, self.index].mean()
        return self.weight * loss + self.regularize(inputs)


class DeepDream(Objective):
    def __init__(self, module, **kwargs):
        super().__init__([module], **kwargs)

    def forward(self, inputs):
        return self.t(0).square().mean() + self.regularize(inputs)