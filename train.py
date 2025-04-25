import os
import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# import optuna
# from optuna.integration import PyTorchLightningPruningCallback
from sklearn.metrics import accuracy_score

from collections import Counter
import string

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Build vocabulary
def build_vocab(texts, max_size=20000):
    word_counts = Counter()
    for text in texts:
        word_counts.update(preprocess_text(text))
    vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common(max_size))}
    vocab['<unk>'] = 0
    return vocab

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = preprocess_text(text)
        indices = [self.vocab.get(token, 0) for token in tokens[:self.max_len]]
        
        # Pad if necessary
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        return torch.tensor(label, dtype=torch.long), torch.tensor(indices, dtype=torch.long)

# Load data from CSV files
def load_data(train_csv, test_csv, text_col='Description', label_col='Class Index'):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Combine 'Title' and 'Description' for better context (optional)
    train_df['combined_text'] = train_df['Title'] + ' ' + train_df['Description']
    test_df['combined_text'] = test_df['Title'] + ' ' + test_df['Description']
    
    # Use combined text or just Description
    text_col = 'combined_text'  # or 'Description' if you prefer
    
    # Build vocabulary
    vocab = build_vocab(train_df[text_col])
    
    # Convert class indices to 0-based (assuming they start at 1)
    train_df[label_col] = train_df[label_col] - 1
    test_df[label_col] = test_df[label_col] - 1
    
    num_classes = len(train_df[label_col].unique())
    
    # Create datasets
    train_dataset = TextDataset(train_df[text_col].values, train_df[label_col].values, vocab)
    test_dataset = TextDataset(test_df[text_col].values, test_df[label_col].values, vocab)
    
    return train_dataset, test_dataset, vocab, num_classes

# Example usage - replace with your actual file paths
train_csv = '/home/user/MLOps/kaggle/input/ag-news/ag_news_train.csv'
test_csv = '/home/user/MLOps/kaggle/input/ag-news/ag_news_test.csv'
train_dataset, test_dataset, vocab, num_classes = load_data(train_csv, test_csv)

# Split train into train/val
train_len = int(0.95 * len(train_dataset))
train_dataset, val_dataset = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Lightning Module
class TextClassificationModel(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        return self.fc(hn[-1])

    def training_step(self, batch, batch_idx):
        labels, texts = batch
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
            labels, texts = batch
            outputs = self(texts)
            loss = self.loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)
            return {"val_loss": loss, "val_acc": acc}
        
    def test_step(self, batch, batch_idx):
        labels, texts = batch
        outputs = self(texts)
        test_loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
def create_dataloaders():
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,  # Increased from default
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,  # Increased from default
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,  # Increased from default
        persistent_workers=True
    )
    return train_loader, val_loader, test_loader

# Define fixed hyperparameters (you can adjust these)
EMBED_DIM = 100
HIDDEN_DIM = 128
LR = 0.001
EPOCHS = 10

# Initialize everything
train_loader, val_loader, test_loader = create_dataloaders()

model = TextClassificationModel(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    output_dim=num_classes,
    lr=0.001
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    save_top_k=1,
    dirpath="checkpoints",
    filename="best_model"
)

trainer = pl.Trainer(
    max_epochs=10,
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