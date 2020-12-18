from functools import partial
from contextlib import AbstractContextManager


import torch

from lucent.utils import progress_bar


class T(AbstractContextManager):
    def __init__(self, modules):
        self.hooks = []
        if isinstance(modules, list):
            modules = {i: x for i, x in enumerate(modules)}
        for name, mod in modules.items():
            self.hooks.append(mod.register_forward_hook(partial(self._hook_fn, name)))
        self.outputs = {}

    def _hook_fn(self, name, module, inp, output):
        self.outputs[name] = output

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self, key):
        return self.outputs[key]


def optimize(
    model, objective, img, steps=20, lr=0.05, weight_decay=1e-6, progress=progress_bar()
):
    optimizer = torch.optim.Adam([img], lr=lr, weight_decay=weight_decay)

    for i in progress(range(steps)):
        model(img)
        objective(img).backward()
        optimizer.step()
        optimizer.zero_grad()

    return img