from torch.nn import Module as NNModule


class BaseDecoder(NNModule):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, **kwargs):
        pass