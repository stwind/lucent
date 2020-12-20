import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, t, idx, name=0, sign=-1):
        super().__init__()
        self.t = t
        self.idx = idx
        self.name = name
        self.sign = sign

    def forward(self, inputs):
        return self.sign * self.t(self.name)[:, self.idx].mean()


class Direction(nn.Module):
    def __init__(self, t, vec, name=0):
        super().__init__()
        self.t = t
        self.vec = vec.reshape(1, -1, 1, 1)

    def forward(self, inputs):
        vec = self.vec.to(inputs.device)
        return torch.sum(self.t(0) * vec, 1).mean()


class DeepDream(nn.Module):
    def __init__(self, t, name=0):
        super().__init__()
        self.t = t
        self.name = name

    def forward(self, inputs):
        return self.t(0).square().mean()