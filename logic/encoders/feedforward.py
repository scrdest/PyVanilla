import torch
import torch.nn as NN

from logic.abstract_defines.abcs import AbstractEncoder
from logic.constants import *

from functools import wraps


class FfEncoder(AbstractEncoder):
    def __init__(self, input_sizes, output_size, enc_depth=1, bottlenecking=1.15, verbose=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = {}

        enc_inp_size = 1
        for x in input_sizes: enc_inp_size *= x
        enc_out_size = None

        if verbose > 0: print('Encoder (linear): ')

        for i in range(1, 1 + enc_depth):
            enc_out_size = int(max(output_size, enc_inp_size // bottlenecking)) if i < enc_depth else output_size
            if verbose > 0: print(i, enc_out_size)
            if enc_inp_size == output_size and i + 1 < enc_depth:
                print("Warning: reached minimum encoder size!")

            newenc = NN.Linear(enc_inp_size, enc_out_size, bias=True)
            NN.init.normal_(newenc.weight, 0, 1)
            NN.init.normal_(newenc.bias, 0, 1)

            newlinenc = (newenc, ())
            self.layers.setdefault(i, []).append(newlinenc)

            enc_inp_size = enc_out_size

        self.out_size = enc_out_size


    def forward(self, data, layer_norm=False, injective_noise=0.05, activation=NN.functional.relu, **kwargs):
        encoding = data.flatten()

        for idx, enc_layer in enumerate(self.layers.values()):

            for layer_spec in enc_layer:

                if idx < max(self.encoder.keys()):
                    enc_func, enc_params = layer_spec
                    encoding = enc_func(encoding, *enc_params)

                    if enc_func not in self.PREPROCESSING_FUNCS:

                        if layer_norm:
                            encoding = (
                                (encoding - encoding.mean()) / encoding.std()
                            )

                        if injective_noise > 0:
                            encoding = (
                                encoding + (
                                    injective_noise * torch.normal(
                                        mean=torch.zeros_like(encoding),
                                        std=torch.ones_like(encoding)
                                    )
                                )
                            )

                        if activation:
                            encoding = activation(encoding)

        return encoding


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
            decorated = constructor(*constructor_args, **constructor_kwargs)
            return decorated

        return _wrapper

    return _parameterized_deco
