from functools import partial


class T(object):
    def __init__(self, modules):
        self.hooks = []
        if isinstance(modules, list):
            modules = {i: x for i, x in enumerate(modules)}
        for name, mod in modules.items():
            self.hooks.append(mod.register_forward_hook(partial(self._hook_fn, name)))
        self.outputs = {}

    def _hook_fn(self, name, module, inp, output):
        self.outputs[name] = output

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self, key):
        return self.outputs[key]