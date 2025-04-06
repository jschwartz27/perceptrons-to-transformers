import torch
import torch.nn as nn
import torch.optim as optim

from helpers.io_helpers import plot


# Define the perceptron
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # 2 inputs -> 1 output

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Add sigmoid for binary output


def main_p(logic_gate_values, learning_rate: float) -> None:

    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = logic_gate_values

    # Initialize model, loss function, optimizer
    model = Perceptron()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Test the model
    with torch.no_grad():
        predictions = model(X)
        print("Inputs:\n", X)
        print("Predicted Outputs:\n", predictions.round())  # Rounded for binary

    losses = list()
    # Train the model
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # Track loss

    # Test the model
    with torch.no_grad():
        predictions = model(X)
        print("Inputs:\n", X)
        print("Predicted Outputs:\n", predictions.round())  # Rounded for binary

    plot(losses=losses)
