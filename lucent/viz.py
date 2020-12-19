from functools import partial
from contextlib import AbstractContextManager
from fastprogress.fastprogress import progress_bar

import torch


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


def optimize(model, optimizer, img, calc_grad, epochs=256):
    for i in progress_bar(range(epochs)):
        model(img)

        calc_grad(img)

        optimizer.step()
        optimizer.zero_grad()

    return img


def calc_by_objective(objective):
    return lambda img: objective(img).backward()


def calc_by_masked_objectives(objectives):
    def inner(img):
        n = len(objectives)
        grads = []
        for i in range(n):
            obj, mask = objectives[i]
            retain_graph = i < n - 1
            obj(img).backward(retain_graph=retain_graph)
            grads.append(img.grad * mask)
            img.grad.zero_()
        g = sum(grads)
        g /= g.std()
        img.grad = g

    return inner