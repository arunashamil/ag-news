import torch


class LSTMClassifier(torch.nn.Module):
    """LSTM model for text classification"""

    def __init__(self, vocab_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 100)
        self.lstm = torch.nn.LSTM(
            input_size=100,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
