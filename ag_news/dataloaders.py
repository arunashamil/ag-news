import json
from collections import Counter

import pandas as pd
import torch
from constants import BATCH_SIZE, VAL_PART, X_INIT_LABEL, X_LABEL, Y_LABEL
from preprocessing import get_tokenized_sentences, pad_num_sentences, preprocessing, tk
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.vocab import Vocab


# Dataset class
class TextDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return torch.tensor(self.sentences[index], dtype=torch.long), torch.tensor(
            self.labels[index], dtype=torch.long
        )


# Load data
def get_dataloaders_after_preprocess(train_df, vocab_path):
    # Apply function to both dataframes
    train_df[X_LABEL] = train_df[X_INIT_LABEL].apply(preprocessing)

    dataset = TextDataset(train_df[X_LABEL], train_df[Y_LABEL])

    # Define split sizes
    train_size = int(VAL_PART * len(dataset))
    val_size = len(dataset) - train_size

    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    x_train = pd.DataFrame(
        {X_LABEL: [dataset.sentences[i] for i in train_dataset.indices]}
    )
    y_train = pd.DataFrame({Y_LABEL: dataset.labels[train_dataset.indices]})

    x_val = pd.DataFrame({X_LABEL: [dataset.sentences[i] for i in val_dataset.indices]})
    y_val = pd.DataFrame({Y_LABEL: dataset.labels[val_dataset.indices]})

    x_train.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)

    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    all_sentences = train_df["preprocessed_sentences"].values.tolist()

    # First get all tokens from your sentences
    token_counter = Counter()
    for tokens in get_tokenized_sentences(all_sentences):
        token_counter.update(tokens)

    # Manually add special tokens
    special_tokens = ["<unk>"]  # Add any other special tokens you need here

    # Create vocabulary
    vocab = Vocab(token_counter, specials=special_tokens)

    # Set default index for unknown words
    vocab.stoi.default_factory = lambda: vocab.stoi["<unk>"]

    # Get entire dictionary which contains key,value pairs of string/ints
    vocab_dict = vocab.stoi

    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)

    x_train_sequences = [
        [vocab_dict[token] for token in tk(text)] for text in x_train[X_LABEL]
    ]
    x_val_sequences = [
        [vocab_dict[token] for token in tk(text)] for text in x_val[X_LABEL]
    ]

    # A few cells ago, decision was made not to use max length, but a value closer to the range of most num sentences.
    max_padding_len = 90

    # cs = current sequence
    x_train_padded = [
        pad_num_sentences(cs, max_padding_len) for cs in x_train_sequences
    ]
    x_val_padded = [pad_num_sentences(cs, max_padding_len) for cs in x_val_sequences]

    y_train_np = y_train[Y_LABEL].values.astype("int64").flatten()
    y_val_np = y_val[Y_LABEL].values.astype("int64").flatten()

    max_of_y = max(set(y_train_np))

    y_train_np[y_train_np == max_of_y] = 0
    y_val_np[y_val_np == max_of_y] = 0

    train_data = TextDataset(x_train_padded, y_train_np)
    val_data = TextDataset(x_val_padded, y_val_np)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    return vocab, train_loader, val_loader


def get_test_dataloader_after_preprocess(test_df, vocab_path):
    # Apply function to both dataframes
    test_df[X_LABEL] = test_df[X_INIT_LABEL].apply(preprocessing)

    test_dataset = TextDataset(test_df[X_LABEL], test_df[Y_LABEL])

    x_test = pd.DataFrame({X_LABEL: test_dataset.sentences})
    y_test = pd.DataFrame({Y_LABEL: test_dataset.labels})

    x_test.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)

    y_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Load vocabulary from json
    f = open(vocab_path)

    vocab_dict = json.load(f)

    # x_test_sequences = [[vocab_dict[token] for token in tk(text)] for text in x_test[X_LABEL]]
    x_test_sequences = [
        [vocab_dict.get(token, vocab_dict["<unk>"]) for token in tk(text)]
        for text in x_test[X_LABEL]
    ]
    # A few cells ago, decision was made not to use max length, but a value closer to the range of most num sentences.
    max_padding_len = 90

    # cs = current sequence
    x_test_padded = [pad_num_sentences(cs, max_padding_len) for cs in x_test_sequences]

    y_test_np = y_test[Y_LABEL].values.astype("int64").flatten()

    max_of_y = max(set(y_test_np))

    y_test_np[y_test_np == max_of_y] = 0

    test_data = TextDataset(x_test_padded, y_test_np)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return test_loader
