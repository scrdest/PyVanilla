from torch.nn import Module as NNModule


class BasePipeline(NNModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, data, *args, **kwargs):
        raise NotImplementedError

    def get_loss(self, inputs, outputs, *args, **kwargs):
        raise NotImplementedError
