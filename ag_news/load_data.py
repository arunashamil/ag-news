import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess import preprocess_text, build_vocab

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

def load_test_data(test_csv, text_col='Description', label_col='Class Index'):

    test_df = pd.read_csv(test_csv)
    
    # Combine 'Title' and 'Description' for better context (optional)
    test_df['combined_text'] = test_df['Title'] + ' ' + test_df['Description']
    
    # Use combined text or just Description
    text_col = 'combined_text'  # or 'Description' if you prefer
    
    # Build vocabulary
    vocab = build_vocab(test_df[text_col])
    
    # Convert class indices to 0-based (assuming they start at 1)
    test_df[label_col] = test_df[label_col] - 1
    
    # Create datasets
    test_dataset = TextDataset(test_df[text_col].values, test_df[label_col].values, vocab)
    
    return test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset):
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