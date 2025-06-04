import re

import torch
from torchtext.data.utils import get_tokenizer

tk = get_tokenizer("basic_english")


def preprocessing(sent):
    """
    Preprocesses a sentence

    Args:
        sent (str): a sentence to be preprocessed

    Returns:
        str: preprocessed sentence
    """

    prepro_sent = sent.lower()
    prepro_sent = re.sub(r"[^\w\s]", "", prepro_sent)
    prepro_sent = tk(prepro_sent)
    prepro_sent = " ".join(prepro_sent)

    return prepro_sent


def get_tokenized_sentences(sentences):
    """
    Tokenizes sentences

    Args:
        sentences (list): sentences to be tokenized

    Yields:
        list[str]: list of string tokens for each sentence
    """
    for sent in sentences:
        yield tk(sent)


def pad_num_sentences(num_sent, max_pad_length):
    """
    Pads tokenized sentences

    Args:
        num_sent (list): tokenized sentence
        max_pad_length (int): controls padding length

    Returns:
        torch.tensor: tokenized sentence after padding
    """
    return torch.nn.functional.pad(
        torch.tensor(num_sent, dtype=torch.int64),
        (0, max_pad_length - len(num_sent)),
        mode="constant",
        value=0,
    )
