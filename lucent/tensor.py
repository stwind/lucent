from contextlib import AbstractContextManager


class Hooks(AbstractContextManager):
    def __init__(self, tensor, funcs):
        self.hooks = []
        for f in funcs:
            self.hooks.append(tensor.register_hook(f))

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()
