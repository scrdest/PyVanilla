import typing
# from typing import Protocol

class Protocol:
    pass

class EncoderProtocol(Protocol):

    def forward(self, *inputs, **kwargs):
        pass

    def get_loss(self, *args, **kwargs) -> typing.Union[float, typing.Dict[typing.Hashable, float]]:
        pass


class DecoderProtocol(Protocol):

    def forward(self, *inputs, **kwargs):
        pass

    def get_loss(self, *args, **kwargs) -> typing.Union[float, typing.Dict[typing.Hashable, float]]:
        pass


class SamplerProtocol(Protocol):

    def forward(self, *inputs, **kwargs):
        pass

    def get_loss(self, *args, **kwargs) -> typing.Union[float, typing.Dict[typing.Hashable, float]]:
        pass


class PipelineProtocol(Protocol):

    def forward(self, *inputs, **kwargs):
        pass

    def get_loss(self, *args, **kwargs) -> typing.Union[float, typing.Dict[typing.Hashable, float]]:
        pass


class ModelProtocol(Protocol):

    def forward(self, *inputs, **kwargs):
        pass

    def get_loss(self, *args, **kwargs) -> typing.Union[float, typing.Dict[typing.Hashable, float]]:
        pass