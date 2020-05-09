import pytest

import logic.abstract_defines.protocols as protocols
import logic.decoders.feedforward
from logic.constants import *


def test_basic_instantiation():
    instance = logic.decoders.feedforward.FfDecoder(
        input_size=10
    )
    assert instance
    # assert isinstance(instance, protocols.DecoderProtocol)


def test_deco_instantiation():
    def kwargreader(**kwargs): return kwargs[PIPE_ARG_DECODER]

    builder = logic.decoders.feedforward.with_ff_decoder(
        input_size=10
    )(kwargreader)

    instance = builder()
    assert instance


if __name__ == '__main__':
    pytest.main([__file__])

