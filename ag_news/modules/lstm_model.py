import torch


class LSTMClassifier(torch.nn.Module):
    """LSTM model for text classification"""

    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        # self.save_hyperparameters()
        self.embedding = torch.nn.Embedding(vocab_size, 100)
        self.lstm = torch.nn.LSTM(100, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
