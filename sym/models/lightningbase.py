import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from ..utils import utils

class Model(pl.LightningModule):
    def __init__(self, model, criterion=None, lr=1e-3, optim="Adam",data_cov=None,zeroparams=None):
        super().__init__()
        self.model = model
        self.Loss = criterion if criterion is not None else MSE(model.out_channels)
        self.optim = optim
        self.zeroparams = zeroparams
        if optim == "Adam":
            self.lr = lr
        else:
            self.lr = self.optim.defaults["lr"]
        self.data_cov = data_cov
        try:
            nfilters = str(model.nfilters)
        except:
            nfilters = "N/A"

        self.hparams["params"] = sum([x.size().numel()
                                      for x in self.model.parameters()])
        self.hparams["nfilters"] = nfilters
        self.hparams["loss"] = self.Loss
        self.hparams["optim"] = optim.__repr__().replace("\n","")
        self.hparams["model"] = self.model.__repr__().replace("\n", "")
        self.hparams["lr"] = self.lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.Loss(yhat, y)
        self.log('train_loss', loss)
        if self.zeroparams is not None:
            self.log("zero_params", sum([torch.sum(abs(g)<self.zeroparams)  for g in self.model.parameters()]))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.Loss(yhat, y)
        if self.model.out_channels == 1:
            if self.model.readout_activation is None:
                preds = torch.sigmoid(yhat.view(-1,1))>.5
            else:
                preds = yhat.view(-1,1)>.5
            y = y.view(-1,1)
        else:
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

        if self.data_cov is not None:
            filters = next(self.model.parameters()).detach().cpu()
            cov_filters = utils.cov_matrix(filters)
            misalignment = utils.Misalignment2(self.data_cov,cov_filters)
            metrics["misalignment"] = misalignment
        if self.zeroparams is not None:
            metrics["zero_params"] = sum([torch.sum(abs(g)<self.zeroparams)  for g in self.model.parameters()])
        self.log_dict(metrics, prog_bar=True,logger=True,on_epoch=True,on_step=False)
        self.logger.log_hyperparams(self.hparams,metrics=metrics)
        return

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self,learning_rate=1e-3):
        if self.optim != "adam":
            optimizer = self.optim
            for g in optimizer.param_groups:
                g["lr"] = learning_rate
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer


class MSE():
    def __init__(self,ntargets):
        self.ntargets = ntargets
    def __call__(self,preds,targets):
        if self.ntargets == 1:
            preds = torch.sigmoid(preds).view(-1)
            targets = targets.view(-1)
        else:
            preds = torch.nn.functional.softmax(preds, dim=-1)
            targets = torch.nn.functional.one_hot(targets, self.ntargets)
        loss = ((preds-targets)**2).mean()
        return loss
    def __repr__(self):
        return f"MSE ntargets{self.ntargets}" 
