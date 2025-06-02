import torch


class RNNClassifier(torch.nn.Module):
    """RNN model for text classification"""

    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 100)
        self.rnn = torch.nn.RNN(100, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.rnn(x)
        return self.fc(hn[-1])
