from typing import Any

import torch

from ag_news.modules.lstm_model import LSTMClassifier
from ag_news.modules.rnn_model import RNNClassifier


def get_model(vocab_size: int, conf: Any) -> torch.nn.Module:
    """Model selection"""

    label = conf["model"]["label"]

    if label == "LSTM":
        return LSTMClassifier(
            vocab_size=vocab_size,
            num_classes=conf["model"]["num_classes"],
            dropout=conf["training"]["dropout"],
        )

    if label == "RNN":
        return RNNClassifier(
            vocab_size=vocab_size,
            num_classes=conf["model"]["num_classes"],
            dropout=conf["training"]["dropout"],
        )
