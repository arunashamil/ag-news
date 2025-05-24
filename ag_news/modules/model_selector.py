from typing import Any, Dict

import torch
from ag_news.modules.lstm_model import LSTMClassifier
from ag_news.modules.rnn_model import RNNClassifier


def get_model(vocab_size: int, model_conf: Dict[str, Any]) -> torch.nn.Module:
    """Model selection"""

    label = model_conf["label"]

    if label == "LSTM":
        return LSTMClassifier(
            vocab_size=vocab_size, num_classes=model_conf["num_classes"]
        )

    if label == "RNN":
        return RNNClassifier(
            vocab_size=vocab_size, num_classes=model_conf["num_classes"]
        )
