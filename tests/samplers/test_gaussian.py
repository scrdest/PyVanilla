import pytest

import logic.samplers.gaussian
from logic.constants import *


def test_basic_instantiation():
    instance = logic.samplers.gaussian.IsoGaussVISampler(
        input_size=10,
        latent_dims=2
    )
    assert instance


def test_deco_instantiation():
    def kwargreader(**kwargs): return kwargs[PIPE_ARG_SAMPLER]

    builder = logic.samplers.gaussian.with_isogauss_sampler(
        input_size=10,
        latent_dims=2
    )(kwargreader)

    instance = builder()
    assert instance


if __name__ == '__main__':
    pytest.main([__file__])
