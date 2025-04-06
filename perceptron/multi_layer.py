import torch
import torch.nn as nn
import torch.optim as optim

from helpers.io_helpers import plot


# Define a simple 2-layer neural network
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # hidden layer with 2 neurons
        self.output = nn.Linear(2, 1)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # hidden activation
        x = torch.sigmoid(self.output(x))  # output activation
        return x


def multi_layer(learning_rate: float) -> None:

    # XOR dataset
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # Initialize
    model = XORNet()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = list()
    # Train
    for _ in range(10000):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # Track loss

    # Test
    with torch.no_grad():
        predictions = model(X)
        print("Predictions:\n", predictions.round())

    plot(losses=losses)
