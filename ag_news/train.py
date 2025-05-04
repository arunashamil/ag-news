import os
import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from collections import Counter

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig

from preprocess import preprocessing
from load_data import TextDataset
from tokenizer import tk, get_tokenized_sentences
from constants import (VAL_PART, LR, EPOCHS, BATCH_SIZE, NUM_WORKERS, X_LABEL, X_INIT_LABEL, Y_LABEL)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def my_train(config : DictConfig) -> None:

    # Load data // TO DO: DOWNLOAD FROM DRIVE 
    train_df = pd.read_csv(config["data_load"]["train_data_path"])
    
    # Apply function to both dataframes
    train_df[X_LABEL] = train_df[X_INIT_LABEL].apply(preprocessing)
    
    dataset = TextDataset(train_df)

    # Define split sizes
    train_size = int(VAL_PART * len(dataset))
    val_size = len(dataset) - train_size

    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    x_train = pd.DataFrame({X_LABEL: [dataset.texts[i] for i in train_dataset.indices]})
    y_train = pd.DataFrame({Y_LABEL: dataset.labels[train_dataset.indices]})

    x_val = pd.DataFrame({X_LABEL: [dataset.texts[i] for i in val_dataset.indices]})
    y_val = pd.DataFrame({Y_LABEL: dataset.labels[val_dataset.indices]})

    x_train.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)

    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    
    x_train_tokens = [tk(sent) for sent in x_train]
    x_val_tokens = [tk(sent) for sent in x_val]

    all_sentences = train_df['preprocessed_sentences'].values.tolist()



    # First get all tokens from your sentences
    token_counter = Counter()
    for tokens in get_tokenized_sentences(all_sentences):
        token_counter.update(tokens)

    # Manually add special tokens
    special_tokens = ['<unk>']  # Add any other special tokens you need here

    # Create vocabulary
    vocab = Vocab(token_counter, specials=special_tokens)

    # Set default index for unknown words
    vocab.stoi.default_factory = lambda: vocab.stoi['<unk>']


    # Get entire dictionary which contains key,value pairs of string/ints
    d = vocab.stoi

    # model = TextClassificationModel(
    #     vocab_size=len(vocab),
    #     embed_dim=100,
    #     hidden_dim=128,
    #     output_dim=num_classes,
    #     lr=LR
    # )

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_acc",
    #     mode="max",
    #     save_top_k=1,
    #     dirpath="checkpoints",
    #     filename="best_model"
    # )

    # trainer = pl.Trainer(
    #     max_epochs=EPOCHS,
    #     callbacks=[checkpoint_callback],
    #     accelerator="auto",
    #     enable_progress_bar=True,
    #     logger=True
    # )

    # # Train and validate
    # trainer.fit(model, train_loader, val_loader)

    # # Get final metrics properly
    # val_results = trainer.validate(model, val_loader)
    # test_results = trainer.test(model, test_loader)

    # print(f"\nFinal Validation Accuracy: {val_results[0]['val_acc']:.4f}")
    # print(f"Final Test Accuracy: {test_results[0]['test_acc']:.4f}")

if __name__ == "__main__":
    my_train()