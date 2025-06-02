import pytorch_lightning as pl
import torch
import torch.nn as nn


class TextClassifier(pl.LightningModule):
    """Module for training and evaluation models
    for text classification task
    """

    def __init__(self, model, lr, vocab_size, dropout):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        texts, labels = batch
        outputs = self(texts)
        test_loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        acc = (preds == labels).float().mean()
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
