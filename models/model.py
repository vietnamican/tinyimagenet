import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torchmetrics

from .base import Base


class Model(Base):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        output = self.backbone(torch.Tensor(1, 3, 64, 64))
        linear_inplanes = output.shape[1]
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(linear_inplanes, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.backbone(x)
        return self.linear(self.dropout(self.flatten(self.adaptive_pool(x))))

    def shared_step(self, x, y):
        logit = self(x)
        return self.loss(logit, y), logit

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, logit = self.shared_step(x, y)
        pred = logit.argmax(dim=1)
        self.train_accuracy.update(pred, y)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, logit = self.shared_step(x, y)
        pred = logit.argmax(dim=1)
        self.val_accuracy.update(pred, y)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_accuracy.compute())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
