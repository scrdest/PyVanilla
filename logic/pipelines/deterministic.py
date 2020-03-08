from logic.constants import *
from logic.pipelines import BasePipeline


class DeterministicPipeline(BasePipeline):
    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, *input, **kwargs):
        encoded = self.encoder.forward(*input)
        decoded = self.decoder.forward(encoded)
        return decoded


def build_deterministic_pipe(*args, **kwargs):
    passed_kwargs = kwargs.copy()

    encoder = passed_kwargs.pop(PIPE_ARG_ENCODER)
    decoder = passed_kwargs.pop(PIPE_ARG_DECODER)

    pipeline = DeterministicPipeline(
        encoder=encoder,
        decoder=decoder,
        *args, **passed_kwargs
    )
    return pipeline
