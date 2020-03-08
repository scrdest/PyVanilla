import torch
import torch.nn as NN

from logic.constants import *
from logic.decoders.base import BaseDecoder

from functools import wraps


class FfDecoder(BaseDecoder):
    def __init__(self, input_size, latent_size, dec_depth=1, upscale=3.5, warn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = {}

        dec_inp_size = int(latent_size)
        dec_out_size = None

        for i in range(1, 1+dec_depth):
            dec_out_size = input_size
            if i < dec_depth:
                dec_out_size = int(min(input_size, dec_inp_size * upscale))
                if warn and dec_out_size == input_size: print("Warning: max decoder size reached!")

            newdec_weight = torch.Tensor(dec_out_size, dec_inp_size)
            NN.init.normal_(newdec_weight, 0, 10)
            newdec_weight_param = NN.Parameter(newdec_weight, requires_grad=True)
            newdec_weight_param.cuda()
            self.weights.append(newdec_weight_param)

            newdec_bias = torch.Tensor(dec_out_size)
            NN.init.normal_(newdec_bias, 0, 10)
            newdec_bias_param = NN.Parameter(newdec_bias, requires_grad=True)
            newdec_bias_param.cuda()
            self.biases.append(newdec_bias_param)

            newdec = (NN.functional.linear, (newdec_weight_param, newdec_bias_param))
            self.layers.setdefault(i, []).append(newdec)

            dec_inp_size = dec_out_size

        self.out_size = dec_out_size


    def forward(self, input, **kwargs):
        pass


def with_ff_decoder(*decoder_args, **decoder_kwargs):

    def _parameterized_deco(constructor):
        _decoder_kws = decoder_kwargs or {}
        decoder = FfDecoder(*decoder_args, **_decoder_kws)

        @wraps(constructor)
        def _wrapper(*raw_args, **raw_kwargs):
            constructor_args = raw_args
            constructor_kwargs = raw_kwargs.copy()
            constructor_kwargs.update({
                PIPE_ARG_DECODER: decoder
            })
            decorated = constructor(*constructor_args, ** constructor_kwargs)
            return decorated

        return _wrapper

    return _parameterized_deco
