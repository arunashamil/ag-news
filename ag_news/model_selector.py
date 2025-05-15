from typing import Any, Dict

import torch
from lstm_model import LSTMClassifier
from rnn_model import RNNClassifier


def get_model(vocab_size: int, model_conf: Dict[str, Any]) -> torch.nn.Module:
    label = model_conf["label"]

    if label == "LSTM":
        return LSTMClassifier(
            vocab_size=vocab_size, num_classes=model_conf["num_classes"]
        )

    if label == "RNN":
        return RNNClassifier(
            vocab_size=vocab_size, num_classes=model_conf["num_classes"]
        )
