import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


class Model(pl.LightningModule):
    def __init__(self, model, criterion=None, lr=1e-3, optim=None):
        super().__init__()
        self.model = model
        self.Loss = criterion if criterion is not None else MSE
        self.optim = optim
        self.lr = lr
        self.hparams["Params"] = sum([x.size().numel()
                                      for x in self.model.parameters()])
        try:
            nfilters = model.nfilters
        except:
            nfilters = None
        self.save_hyperparameters(
            {"nfilters": nfilters, "Model": self.model.__repr__().replace("\n", "")})

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.Loss(yhat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.Loss(yhat, y)
        preds = torch.argmax(yhat, dim=1)
        acc = accuracy(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        metrics = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(metrics, prog_bar=True, logger=True,
                      on_epoch=True, on_step=False)
        try:
            self.logger.log_hyperparams(self.hparams, metrics=metrics)
        except:
            self.logger.log_hyperparams(self.hparams)
        return metrics

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']
        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        metrics = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
#         self.log_dict(metrics, prog_bar=True,logger=True,on_epoch=True,on_step=False)
#         self.logger.log_hyperparams(self.hparams,metrics=metrics)
        return

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self,learning_rate=1e-3):
        if self.optim is not None:
            optimizer = self.optim
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer


def MSE(preds, targets):
    preds = torch.nn.functional.softmax(preds, dim=-1)
    targets = torch.nn.functional.one_hot(targets, 10)
    loss = ((preds-targets)**2).mean()
    return loss
