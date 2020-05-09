import sys
import typing

import logging

logging.getLogger(__name__).addHandler(logging.StreamHandler(sys.stdout))

import pytorch_lightning as lightning
import torch

NNModule = lightning.LightningModule

import torchvision.transforms as transforms

mnist_transf = transforms.Compose([
    transforms.ToTensor()
])

import torch.utils.data.dataloader as loader

import logic.abstract_defines.abcs as custom_abcs
from logic.utils import *


class ModularVAE(NNModule):
    def __init__(
        self,
        pipelines: typing.List[custom_abcs.AbstractPipeline],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pipelines = pipelines


    def forward(self, *input, **kwargs):
        pipe_inputs = {}
        pipe_outputs = {}
        current_output = input
        for pipeline in self.pipelines:
            pipe_inputs[pipeline] = current_output
            current_output = pipeline.forward(*current_output)
            pipe_outputs[pipeline] = current_output
        return current_output, pipe_inputs, pipe_outputs


    def get_loss(self, inputs_per_pipe, outputs_per_pipe):
        loss_per_pipe = {}
        for pipeline in self.pipelines:
            pipe_inp = inputs_per_pipe.get(pipeline)
            pipe_out = outputs_per_pipe.get(pipeline)
            loss_per_pipe[pipeline] = pipeline.get_loss(pipe_inp, pipe_out)


    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        outputs = {}
        losses = {}

        for pipeline in self.pipelines:
            pipeline_output = pipeline.forward(x)
            pipeline_loss = pipeline.get_loss(x, y, pipeline_output)
            outputs[pipeline] = pipeline_output
            losses[pipeline] = pipeline_loss

        total_loss = sum(losses.values())
        # total_reconstruction_losses = {}
        # total_divergence_losses = {}


        tensorboard_logs = {
            'train_total_loss': total_loss
            # 'train_reconstruction_loss': losses['reconstruction'],
            # 'train_divergence_loss': losses['latent']
        }

        return_dict = {
            'loss': total_loss,
            'log': tensorboard_logs
        }

        return return_dict


    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        outputs = {}
        losses = {}

        for pipeline in self.pipelines:
            pipeline_output = pipeline.forward(x)
            pipeline_loss = pipeline.get_loss(x, y, pipeline_output)
            outputs[pipeline] = pipeline_output
            losses[pipeline] = pipeline_loss

        total_loss = sum(losses.values())
        return_dict = {
            'val_loss': total_loss,
        }
        return return_dict


    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.003, amsgrad=True)
        #optim = torch.optim.SGD(self.parameters(), lr=0.001)
        return optim


    @lightning.data_loader
    def train_dataloader(self):
        dataset = self.dataset
        data_tr = self.dataset_transform
        train_loc = self.data_locs.get('train') or localdir('train')
        return loader.DataLoader(dataset(train_loc, train=True, download=True, transform=data_tr), batch_size=5)


    @lightning.data_loader
    def val_dataloader(self):
        dataset = self.dataset
        data_tr = self.dataset_transform
        val_loc = self.data_locs.get('val') or localdir('val')
        return loader.DataLoader(dataset(val_loc, train=True, download=True, transform=data_tr), batch_size=5)


    @lightning.data_loader
    def test_dataloader(self):
        dataset = self.dataset
        data_tr = self.dataset_transform
        test_loc = self.data_locs.get('test') or localdir('test')
        return loader.DataLoader(dataset(test_loc, train=False, download=True, transform=data_tr), batch_size=5)




