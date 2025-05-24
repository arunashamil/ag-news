import re
from typing import Iterator

import torch
from torchtext.data.utils import get_tokenizer

tk = get_tokenizer("basic_english")


def preprocessing(sent: str) -> str:
    """Preprocesses a sentence

    Args:
        sent (str): a sentence to be preprocessed

    Returns:
        prepro_sent (str): preprocessed sentence
    """

    prepro_sent = sent.lower()
    prepro_sent = re.sub(r"[^\w\s]", "", prepro_sent)
    prepro_sent = tk(prepro_sent)
    prepro_sent = " ".join(prepro_sent)

    return prepro_sent


def get_tokenized_sentences(sentences: list) -> Iterator[list[str]]:
    """Tokenizes sentences

    Args:
        sentences (list): sentences to be tokenized

    Yields:
        tk(sent) (list[str]): list of string tokens for each sentence
    """
    for sent in sentences:
        yield tk(sent)


def pad_num_sentences(num_sent, max_pad_length):
    return torch.nn.functional.pad(
        torch.tensor(num_sent, dtype=torch.int64),
        (0, max_pad_length - len(num_sent)),
        mode="constant",
        value=0,
    )
