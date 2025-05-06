import torch
from torch.utils.data import Dataset

# Dataset class
class TextDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return torch.tensor(self.sentences[index], dtype=torch.long), \
            torch.tensor(self.labels[index], dtype=torch.long)
