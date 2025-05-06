import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig

from preprocess import get_dataloaders_after_preprocess
from trainer import TextClassificationModel
from constants import (NUM_CLASSES, LR, EPOCHS)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def my_train(config : DictConfig) -> None:

    train_df = pd.read_csv(config["data_load"]["train_data_path"])

    vocab, train_loader, val_loader = get_dataloaders_after_preprocess(train_df)
    vocab_size = len(vocab.stoi) + 1

    model = TextClassificationModel(
        vocab_size=vocab_size,
        embed_dim=100,
        hidden_dim=128,
        output_dim=NUM_CLASSES,
        lr=LR
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=config["model"]["model_local_path"],
        filename="model_{val_acc:.2f}"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        enable_progress_bar=True,
        logger=True
    )

    # Train and validate
    trainer.fit(model, train_loader, val_loader)

    # Get final metrics properly
    val_results = trainer.validate(model, val_loader)
    # test_results = trainer.test(model, test_loader)

    # print(f"\nFinal Validation Accuracy: {val_results[0]['val_acc']:.4f}")
    # # print(f"Final Test Accuracy: {test_results[0]['test_acc']:.4f}")

if __name__ == "__main__":
    my_train()