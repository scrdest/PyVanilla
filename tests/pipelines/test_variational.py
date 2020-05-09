import pytest

import logic.pipelines.variational

from logic.constants import *
import logic.abstract_defines.protocols as protocols

from logic.encoders.feedforward import with_ff_encoder, FfEncoder
from logic.samplers.gaussian import with_isogauss_sampler, IsoGaussVISampler
from logic.decoders.feedforward import with_ff_decoder, FfDecoder


def test_decorator_injection():

    @with_ff_encoder(input_sizes=[10], output_size=2)
    @with_isogauss_sampler(input_size=10, latent_dims=2)
    @with_ff_decoder(input_size=10)
    def ff_builder(*args, **kwargs):
        return logic.pipelines.variational.build_vi_pipeline(*args, **kwargs)

    built = ff_builder()
    assert built
    # assert isinstance(built, protocols.PipelineProtocol)


def test_encoder_attached():

    @with_ff_decoder(input_size=10)
    @with_isogauss_sampler(input_size=10, latent_dims=2)
    def ff_builder(inj_encoder, *args, **kwargs):
        injected_kwargs = kwargs.copy()
        injected_kwargs[PIPE_ARG_ENCODER] = inj_encoder
        return logic.pipelines.variational.build_vi_pipeline(*args, **injected_kwargs)

    encoder = FfEncoder(input_sizes=[10], output_size=2)
    built = ff_builder(encoder)
    assert built
    assert encoder is built.encoder
    # assert isinstance(built, protocols.PipelineProtocol)
    # assert isinstance(encoder, protocols.EncoderProtocol)


def test_decoder_attached():

    @with_ff_encoder(input_sizes=[10], output_size=2)
    @with_isogauss_sampler(input_size=10, latent_dims=2)
    def ff_builder(inj_decoder, *args, **kwargs):
        injected_kwargs = kwargs.copy()
        injected_kwargs[PIPE_ARG_DECODER] = inj_decoder
        return logic.pipelines.variational.build_vi_pipeline(*args, **injected_kwargs)

    decoder = FfDecoder(input_size=10)
    built = ff_builder(decoder)
    assert built
    assert decoder is built.decoder
    # assert isinstance(built, protocols.PipelineProtocol)
    # assert isinstance(decoder, protocols.DecoderProtocol)


def test_sampler_attached():

    @with_ff_encoder(input_sizes=[10], output_size=2)
    @with_ff_decoder(input_size=10)
    def ff_builder(inj_sampler, *args, **kwargs):
        injected_kwargs = kwargs.copy()
        injected_kwargs[PIPE_ARG_SAMPLER] = inj_sampler
        return logic.pipelines.variational.build_vi_pipeline(*args, **injected_kwargs)

    sampler = IsoGaussVISampler(input_size=10, latent_dims=2)
    built = ff_builder(sampler)
    assert built
    assert sampler is built.latent_sampler
    # assert isinstance(built, protocols.PipelineProtocol)
    # assert isinstance(sampler, protocols.SamplerProtocol)


if __name__ == '__main__':
    pytest.main([__file__])
