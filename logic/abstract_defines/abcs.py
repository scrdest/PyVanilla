from torch.nn import Module as NNModule

import logic.utils


class AbstractDecoder(NNModule):
    PREPROCESSING_FUNCS = {
        logic.utils.flat2matrix,
        logic.utils.matrix2flat
    }

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        pass

    def get_loss(self, *args, **kwargs):
        pass


class AbstractEncoder(NNModule):
    PREPROCESSING_FUNCS = {
        logic.utils.flat2matrix,
        logic.utils.matrix2flat
    }

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        pass

    def get_loss(self, *args, **kwargs):
        pass


class AbstractVISampler(NNModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        pass

    def get_loss(self, *args, **kwargs):
        pass


class AbstractPipeline(NNModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        """Kleisli-composes forward steps from each component together,
        i.e. Pipeline(A,B,C).forward() == C.forward(B.forward(A.forward()))
        with any glue between the values and side-effects included as necessary.
        """
        raise NotImplementedError

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError
