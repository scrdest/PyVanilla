from torch.nn import Module as NNModule


class BasePipeline(NNModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *input, **kwargs):
        raise NotImplementedError

    def get_loss(self, inputs, outputs):
        raise NotImplementedError
