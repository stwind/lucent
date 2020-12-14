import torch
import torch.nn as nn

from lucent.viz import T


class Objective(nn.Module):
    def __init__(self, modules, regularizers=[]):
        super().__init__()
        self.t = T(modules)
        self.regularizers = regularizers

    def regularize(self, inputs):
        loss = sum([w * reg(inputs) for reg, w in self.regularizers])
        return loss


class Channel(Objective):
    def __init__(self, module, index, sign=-1, **kwargs):
        super().__init__([module], **kwargs)
        self.index = index
        self.sign = sign

    def forward(self, inputs):
        loss = self.t(0)[:, self.index].mean()
        return self.sign * loss + self.regularize(inputs)


class Channels(Objective):
    def __init__(self, channels, sign=-1, **kwargs):
        modules = [mod for mod, _, _ in channels]
        super().__init__(modules, **kwargs)
        self.channels = channels
        self.sign = sign

    def forward(self, inputs):
        loss = sum(
            [
                self.t(i)[:, idx].mean() * w
                for i, (_, idx, w) in enumerate(self.channels)
            ]
        )
        return self.sign * loss.to(inputs.device) + self.regularize(inputs)


class DeepDream(Objective):
    def __init__(self, module, **kwargs):
        super().__init__([module], **kwargs)

    def forward(self, inputs):
        return self.t(0).square().mean() + self.regularize(inputs)


class Direction(Objective):
    def __init__(self, module, vec, **kwargs):
        super().__init__([module], **kwargs)
        self.vec = vec.reshape(1, -1, 1, 1)

    def forward(self, inputs):
        vec = self.vec.to(inputs.device)
        loss = torch.sum(self.t(0) * vec, 1).mean()
        return loss + self.regularize(inputs)