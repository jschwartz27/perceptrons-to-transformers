import matplotlib.pyplot as plt

plt.style.use("dark_background")


def plot(losses) -> None:
    plt.plot(losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
