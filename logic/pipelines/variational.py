from logic.constants import *
from logic.abstract_defines import abcs


class VariationalPipeline(abcs.AbstractPipeline):

    def __init__(
        self,
        encoder: abcs.AbstractEncoder,
        latent_sampler: abcs.AbstractVISampler,
        decoder: abcs.AbstractDecoder,
        *args, **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.latent_sampler = latent_sampler
        self.decoder = decoder


    def forward(self, *input, **kwargs):
        encoded = self.encoder.forward(*input)
        latent = self.latent_sampler.forward(encoded)
        decoded = self.decoder.forward(latent)
        return decoded


    def get_loss(self, inputs, outputs, *args, **kwargs):
        encoder_loss = self.encoder.get_loss()
        decoder_loss = self.decoder.get_loss()
        latent_loss = self.latent_sampler.get_loss()
        total_loss = encoder_loss + decoder_loss + latent_loss
        return total_loss



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

