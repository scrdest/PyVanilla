from torch.nn import Module as NNModule

import logic.utils

class BaseDecoder(NNModule):
    PREPROCESSING_FUNCS = {
        logic.utils.flat2matrix,
        logic.utils.matrix2flat
    }

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, data, *args, **kwargs):
        pass
