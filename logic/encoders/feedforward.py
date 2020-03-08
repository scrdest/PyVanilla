import torch
import torch.nn as NN

from logic.encoders.base import BaseEncoder
from logic.constants import *

from functools import wraps


class FfEncoder(BaseEncoder):
    def __init__(self, inp_size, latent_dim, enc_depth=1, bottlenecking=1.15, verbose=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        enc_inp_size = 1
        for x in inp_size: enc_inp_size *= x
        enc_out_size = None
        if verbose > 0: print('Encoder (linear): ')
        for i in range(1, 1 + enc_depth):
            enc_out_size = int(max(latent_dim, enc_inp_size // bottlenecking)) if i < enc_depth else latent_dim
            if verbose > 0: print(i, enc_out_size)
            if enc_inp_size == latent_dim and i + 1 < enc_depth:
                print("Warning: reached minimum encoder size!")

            newenc_weight = torch.Tensor(enc_out_size, enc_inp_size)
            NN.init.normal_(newenc_weight, 0, 1)

            newenc_weight_param = NN.Parameter(newenc_weight)
            self.weights.append(newenc_weight_param)

            newenc_bias = torch.Tensor(enc_out_size)
            NN.init.normal_(newenc_bias, 0, 1)

            newenc_bias_param = NN.Parameter(newenc_bias)
            self.biases.append(newenc_bias_param)

            newlinenc = (NN.functional.linear, (newenc_weight_param, newenc_bias_param))
            self.encoder.setdefault(i, []).append(newlinenc)

            enc_inp_size = enc_out_size

        self.out_size = enc_out_size


    def forward(self, input, **kwargs):
        pass


def with_ff_encoder(*encoder_args, **encoder_kwargs):
    def _parameterized_deco(constructor):
        _encoder_kws = encoder_kwargs or {}
        encoder = FfEncoder(*encoder_args, **_encoder_kws)

        @wraps(constructor)
        def _wrapper(*raw_args, **raw_kwargs):
            constructor_args = raw_args
            constructor_kwargs = raw_kwargs.copy()
            constructor_kwargs.update({
                PIPE_ARG_ENCODER: encoder
            })
            decorated = constructor(*constructor_args, ** constructor_kwargs)
            return decorated

        return _wrapper

    return _parameterized_deco
