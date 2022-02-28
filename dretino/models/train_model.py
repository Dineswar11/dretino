from datetime import datetime

import pytorch_lightning as pl
from dretino.models.coralloss import ModelCORAL, cal_coral_loss
from dretino.models.cornloss import ModelCORN, cal_corn_loss
from dretino.models.crossentropy import ModelCE, ce_loss
from dretino.models.mseloss import ModelMSE, mse_loss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score
from torchmetrics import CohenKappa


class Model(pl.LightningModule):
    def __init__(self,
                 loss='ce',
                 model_name='resnet50d',
                 num_classes=5,
                 lr=3e-4,
                 num_neurons=512,
                 n_layers=2,
                 dropout_rate=0.2):
        super(Model, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.loss = loss
        self.num_classes = num_classes
        self.lr = lr
        self.model_name = model_name
        self.n_layers = n_layers
        self.num_neurons = num_neurons
        self.dropout_rate = dropout_rate
        self.accuracy = Accuracy()
        self.metric = F1Score(num_classes=self.num_classes)
        self.kappametric = CohenKappa(num_classes=self.num_classes)
        if self.loss == 'ce':
            self.model = ModelCE(self.model_name,
                                 self.num_classes,
                                 self.num_neurons,
                                 self.n_layers,
                                 self.dropout_rate)
        elif self.loss == 'mse':
            self.model = ModelMSE(self.model_name,
                                  self.num_classes,
                                  self.num_neurons,
                                  self.n_layers,
                                  self.dropout_rate)
        elif self.loss == 'corn':
            self.model = ModelCORN(self.model_name,
                                   self.num_classes,
                                   self.num_neurons,
                                   self.n_layers,
                                   self.dropout_rate)
        elif self.loss == 'coral':
            self.model = ModelCORAL(self.model_name,
                                    self.num_classes,
                                    self.num_neurons,
                                    self.n_layers,
                                    self.dropout_rate)
        else:
            s = ('Invalid value for `reduction`. Should be "ce", '
                 '"mse", "corn" or "coral". Got %s' % self.loss)
            raise ValueError(s)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preds, loss, acc, f1_score, kappa_score = self._get_preds_loss_accuracy(batch)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        self.log('train_kappa', kappa_score, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc, f1_score, kappa_score = self._get_preds_loss_accuracy(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        self.log('val_kappa', kappa_score, prog_bar=False, on_epoch=True)
        self.log('valid_F1_score', f1_score, prog_bar=False, on_epoch=True)
        return preds

    def test_step(self, batch, batch_idx):
        _, loss, acc, f1_score, kappa_score = self._get_preds_loss_accuracy(batch)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        self.log('test_F1_score', f1_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

        return {'optimizer': optimizer}

    def _get_preds_loss_accuracy(self, batch):
        x, y = batch
        logits = self(x)
        if self.loss == 'ce':
            loss, preds, y = ce_loss(logits, y)
        elif self.loss == 'mse':
            loss, preds, y = mse_loss(logits, y)
        elif self.loss == 'corn':
            loss, preds, y = cal_corn_loss(logits, y, self.num_classes)
        elif self.loss == 'coral':
            loss, preds, y = cal_coral_loss(logits, y, self.num_classes)
        else:
            s = ('Invalid value for `reduction`. Should be "ce", '
                 '"mse", "corn" or "coral". Got %s' % self.loss)
            raise ValueError(s)

        acc = self.accuracy(preds, y)
        f1_score = self.metric(preds, y)
        kappa_score = self.kappametric(preds, y)
        return preds, loss, acc, f1_score, kappa_score


def train(Model, dm, wab=False, fast_dev_run=False, overfit_batches=False, **kwargs):
    if wab and fast_dev_run:
        s = "Both wab and fast_dev_run cannot be true at the same time"
        raise RuntimeError(s)
    if fast_dev_run and overfit_batches:
        s = "Both overfit_batches and fast_dev_run cannot be true at the same time"
        raise RuntimeError(s)
    num_neurons = kwargs['num_neurons']
    num_layers = kwargs['num_layers']
    dropout = kwargs['dropout']
    lr = kwargs['lr']
    loss = kwargs['loss']
    model_name = kwargs['model_name']

    model = Model(model_name=model_name,
                  loss=loss,
                  num_neurons=num_neurons,
                  n_layers=num_layers,
                  dropout_rate=dropout,
                  lr=lr,
                  num_classes=5)

    file_name = f"{str(datetime.now()).replace(' ','').replace(':','')}_{loss}_{num_neurons}_{num_layers}_{dropout}_{lr}"
    csv_logger = CSVLogger(save_dir="../reports/csv_logs/", name=file_name)
    tensorboard_logger = TensorBoardLogger(save_dir="../reports/tensorboard_logs/", name=file_name)
    if wab:
        wandb_logger = WandbLogger(project=kwargs['project'], log_model=False)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                          dirpath='../models',
                                          save_top_k=1,
                                          save_last=False,
                                          save_weights_only=True,
                                          filename=file_name,
                                          verbose=False,
                                          mode='min')
    trainer = Trainer(gpus=kwargs['gpus'],
                      max_epochs=kwargs['epochs'],
                      callbacks=[checkpoint_callback],
                      logger=[csv_logger,
                              tensorboard_logger])
    if wab:
        wandb_logger.watch(model)
        trainer = Trainer(gpus=kwargs['gpus'],
                          max_epochs=kwargs['epochs'],
                          callbacks=[checkpoint_callback],
                          logger=[csv_logger,
                                  tensorboard_logger,
                                  wandb_logger])
    if fast_dev_run:
        trainer = Trainer(gpus=kwargs['gpus'],
                          fast_dev_run=fast_dev_run)
    if overfit_batches:
        trainer = Trainer(gpus=kwargs['gpus'],
                          overfit_batches=1,
                          max_epochs=kwargs['epochs'],
                          callbacks=[checkpoint_callback],
                          logger=[csv_logger,
                                  tensorboard_logger])
    if wab and overfit_batches:
        trainer = Trainer(gpus=kwargs['gpus'],
                          max_epochs=kwargs['epochs'],
                          callbacks=[checkpoint_callback],
                          logger=[csv_logger,
                                  tensorboard_logger,
                                  wandb_logger])

    trainer.fit(model, dm)
    return file_name, trainer