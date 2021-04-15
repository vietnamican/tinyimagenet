import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import TinyImagenetDataset, transformer
from models import Model, backbone

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

pl.seed_everything(42)

# Data Setup
traindataset = TinyImagenetDataset('sets/random/0/wnids10.txt', 'tiny-imagenet-200/train', transform=transformer['train'])
trainloader = DataLoader(traindataset, batch_size=2,
                         pin_memory=True, num_workers=1)
valdataset = TinyImagenetDataset('sets/random/0/wnids10.txt', 'tiny-imagenet-200/train', transform=transformer['val'])
valloader = DataLoader(valdataset, batch_size=2,
                       pin_memory=True, num_workers=1)

device = 'cpu'

if __name__ == '__main__':

    mode = 'training'
    if mode == 'training':
        log_name = 'tiny_imagenet_logs/{}'.format(mode)
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            name=log_name,
            # log_graph=True,
            # version=0
        )
        loss_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='',
            filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
            save_top_k=-1,
            mode='min',
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks = [loss_callback, lr_monitor]
        model = Model(backbone=backbone.MobileNetV2(
            num_classes=10), num_classes=10)
        # state_dict = model_zoo.load_url(
        #     model_urls['mobilenet_v2'], progress=True)
        # model.backbone.migrate(state_dict, force=True)
        # x = torch.Tensor(1, 3, 64, 64)
        # model(x)
        if device == 'tpu':
            trainer = pl.Trainer(
                max_epochs=90,
                logger=logger,
                callbacks=callbacks,
                tpu_cores=8
            )
        elif device == 'gpu':
            trainer = pl.Trainer(
                max_epochs=90,
                logger=logger,
                callbacks=callbacks,
                gpus=1
            )
        else:
            trainer = pl.Trainer(
                max_epochs=90,
                logger=logger,
                callbacks=callbacks,
                limit_train_batches=0.01,
                limit_val_batches=0.01,
            )
        trainer.fit(model, trainloader, valloader, )
