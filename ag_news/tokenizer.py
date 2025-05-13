import torch
from torchtext.data.utils import get_tokenizer

tk = get_tokenizer("basic_english")


def get_tokenized_sentences(sentences):
    for sent in sentences:
        yield tk(sent)


def pad_num_sentences(num_sent, max_pad_length):
    return torch.nn.functional.pad(
        torch.tensor(num_sent, dtype=torch.int64),
        (0, max_pad_length - len(num_sent)),
        mode="constant",
        value=0,
    )
