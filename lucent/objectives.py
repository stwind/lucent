import torch
import torch.nn as nn

from lucent.tensor import l2_normalize


def maybe_batch(output, batch=None):
    if batch is not None:
        return output[batch : batch + 1]
    return output


class Sum(nn.Module):
    def __init__(self, objs):
        super().__init__()
        self.objs = objs

    def forward(self, inputs):
        return sum([o(inputs) * w for o, w in self.objs])


class Diversity(nn.Module):
    def __init__(self, t, name=0):
        super().__init__()
        self.t = t
        self.name = name

    def forward(self, inputs):
        output = self.t(self.name)
        n, _, _, c = output.size()
        flattened = output.reshape((n, -1, c))
        grams = torch.matmul(flattened.transpose(1, 2), flattened)
        grams = l2_normalize(grams, (1, 2))
        return (
            sum(
                [
                    sum([(grams[i] * grams[j]).sum() for j in range(n) if j != i])
                    for i in range(n)
                ]
            )
            / n
        )


class Channel(nn.Module):
    def __init__(self, t, idx, name=0, sign=-1, batch=None):
        super().__init__()
        self.t = t
        self.idx = idx
        self.name = name
        self.sign = sign
        self.batch = batch

    def forward(self, inputs):
        output = maybe_batch(self.t(self.name), self.batch)
        return self.sign * output[:, self.idx].mean()


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