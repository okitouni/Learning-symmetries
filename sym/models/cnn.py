import torch
import torch.nn as nn


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
    ])[activation]


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, h=280, w=280, nfilters=10, ks=28, activation='relu', *args, **kwargs):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, nfilters, kernel_size=ks,
                      stride=ks, padding=0, bias=True),
            nn.BatchNorm2d(f),
            activation_func(activation)
        )

        self.decoder = nn.Linear(h*w//ks**2*nfilters, out_channels)

    def forward(self, x):
        x = self.conv_block1(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.Loss = torch.nn.CrossEntropyLoss()

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
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self, learning_rate=1e-2):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(trainset, [
                                                                             55000, 5000])
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = testset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, num_workers=6, batch_size=320)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, num_workers=6, batch_size=320)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, num_workers=6, batch_size=320)
