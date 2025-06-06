from ag_news.modules.lstm_model import LSTMClassifier
from ag_news.modules.rnn_model import RNNClassifier


def get_model(vocab_size, conf):
    """Model selection"""

    label = conf["model"]["label"]

    if label == "LSTM":
        return LSTMClassifier(
            vocab_size=vocab_size,
            input_size=conf["model"]["input_size"],
            hidden_size=conf["model"]["hidden_size"],
            num_layers=conf["model"]["num_layers"],
            num_classes=conf["model"]["num_classes"],
            dropout=conf["training"]["dropout"],
        )

    if label == "RNN":
        return RNNClassifier(
            vocab_size=vocab_size,
            input_size=conf["model"]["input_size"],
            hidden_size=conf["model"]["hidden_size"],
            num_layers=conf["model"]["num_layers"],
            num_classes=conf["model"]["num_classes"],
            dropout=conf["training"]["dropout"],
        )
