import torch
from contextlib import AbstractContextManager
from lucent.utils import binomial_filter, lap_normalize


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