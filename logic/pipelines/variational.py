from logic.constants import *
from logic.pipelines import BasePipeline


class VariationalPipeline(BasePipeline):
    def __init__(self, encoder, latent_sampler, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.latent_sampler = latent_sampler
        self.decoder = decoder

    def forward(self, *input, **kwargs):
        encoded = self.encoder.forward(*input)
        latent = self.latent_sampler.forward(encoded)
        decoded = self.decoder.forward(latent)
        return decoded


def build_vi_pipeline(*args, **kwargs):
    passed_kwargs = kwargs.copy()

    encoder = passed_kwargs.pop(PIPE_ARG_ENCODER)
    decoder = passed_kwargs.pop(PIPE_ARG_DECODER)
    sampler = passed_kwargs.pop(PIPE_ARG_SAMPLER)

    pipeline = VariationalPipeline(
        encoder=encoder,
        decoder=decoder,
        latent_sampler=sampler,
        *args, **passed_kwargs
    )
    return pipeline

