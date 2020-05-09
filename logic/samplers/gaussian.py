import torch
import torch.nn as NN

from logic.constants import *
from logic.abstract_defines.abcs import AbstractVISampler

from functools import wraps

from logic.utils import lazy_attach


class IsoGaussVISampler(AbstractVISampler):
    def __init__(self, input_size, latent_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.latent_dims = latent_dims

        self.mean_transform = lazy_attach(input_size, latent_dims)(NN.Linear)
        self.stddev_transform = lazy_attach(input_size, latent_dims)(NN.Linear)


    def encode(self, data, *args, **kwargs):
        mean = self.mean_transform(data)
        log_var = self.stddev_transform(data)
        return mean, log_var


    def sample(self, mean, log_var):
        stddev = torch.exp(0.5 * log_var)
        noise = torch.randn_like(stddev)
        return mean + noise * stddev


    def forward(self, data, *args, **kwargs):
        mean, log_var = self.encode(data)
        z = self.sample(mean, log_var)
        return z, data, mean, log_var


def with_isogauss_sampler(*sampler_args, **sampler_kwargs):

    def _parameterized_deco(constructor):
        _sampler_kws = sampler_kwargs or {}
        sampler = IsoGaussVISampler(*sampler_args, **_sampler_kws)

        @wraps(constructor)
        def _wrapper(*raw_args, **raw_kwargs):
            constructor_args = raw_args
            constructor_kwargs = raw_kwargs.copy()
            constructor_kwargs.update({
                PIPE_ARG_SAMPLER: sampler
            })
            decorated = constructor(*constructor_args, ** constructor_kwargs)
            return decorated

        return _wrapper

    return _parameterized_deco
