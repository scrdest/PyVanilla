import pytest

import logic.encoders.feedforward
from logic.constants import *


def test_basic_instantiation():
    instance = logic.encoders.feedforward.FfEncoder(
        input_sizes=[10],
        output_size=20
    )
    assert instance


def test_deco_instantiation():
    def kwargreader(**kwargs): return kwargs[PIPE_ARG_ENCODER]

    builder = logic.encoders.feedforward.with_ff_encoder(
        input_sizes=[10],
        output_size=20
    )(kwargreader)

    instance = builder()
    assert instance


if __name__ == '__main__':
    pytest.main([__file__])

