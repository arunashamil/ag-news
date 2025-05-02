import os
import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig

from trainer import TextClassificationModel
from load_data import load_data, create_dataloaders

from constants import (VAL_PART, LR, EPOCHS, BATCH_SIZE)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def my_train(config : DictConfig) -> None:

    # Load data from csv files via path
    train_csv = config["data_load"]["train_data_path"]
    test_csv = config["data_load"]["test_data_path"]
    train_dataset, test_dataset, vocab, num_classes = load_data(train_csv, test_csv)

    # Split train into train/val
    train_len = int(VAL_PART * len(train_dataset))
    train_dataset, val_dataset = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # Initialize everything
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    model = TextClassificationModel(
        vocab_size=len(vocab),
        embed_dim=100,
        hidden_dim=128,
        output_dim=num_classes,
        lr=LR
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints",
        filename="best_model"
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
    test_results = trainer.test(model, test_loader)

    print(f"\nFinal Validation Accuracy: {val_results[0]['val_acc']:.4f}")
    print(f"Final Test Accuracy: {test_results[0]['test_acc']:.4f}")

if __name__ == "__main__":
    my_train()