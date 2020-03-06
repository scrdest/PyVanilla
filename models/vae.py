import sys

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
import torchvision.transforms as transforms
import torchvision.datasets.mnist as mnist_datasets

from logic.constants import *
from logic.utils import *


class ModularVAE(NNModule):

    def __init__(self, modules=None, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, *args, **kwargs):
        pass


    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        predicted, sampledists = self.forward(x)
        losses = self.get_loss(x=x, predicted=predicted, expectation=self.expectation, actuals=sampledists)

        tensorboard_logs = {
            'train_total_loss': losses['total'],
            'train_reconstruction_loss': losses['reconstruction'],
            'train_divergence_loss': losses['latent']
        }
        return_dict = {
            'loss': losses['total'],
            'log': tensorboard_logs
        }
        return return_dict


    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        predicted, sampledists = self.forward(x)
        losses = self.get_loss(x=x, predicted=predicted, expectation=self.expectation, actuals=sampledists)
        return_dict = {
            'val_loss': losses['total'],
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




