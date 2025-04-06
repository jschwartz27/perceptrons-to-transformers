import argparse
from enum import Enum
import torch

from perceptron.multi_layer import multi_layer
from perceptron.single_layer import main_p


class Stream(Enum):
    SingleAnd = 0
    SingleXor = 1
    MultiXor = 2


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Type Parameter")
    parser.add_argument("-s", "--stream", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument(
        "-r", "--rate", type=float, choices=range(1), required=False, default=0.1
    )
    return parser.parse_args()


def main(stream: Stream, learning_rate: float) -> None:
    logic_gates = {
        "AND": torch.tensor([[0], [0], [0], [1]], dtype=torch.float32),
        "XOR": torch.tensor([[0], [1], [1], [0]], dtype=torch.float32),
    }

    match stream:
        case Stream.SingleAnd:
            main_p(logic_gate_values=logic_gates["AND"], learning_rate=learning_rate)
        case Stream.SingleXor:
            main_p(logic_gate_values=logic_gates["XOR"], learning_rate=learning_rate)
        case Stream.MultiXor:
            multi_layer(learning_rate=learning_rate)


if __name__ == "__main__":
    a = get_args()
    main(stream=Stream(a.stream), learning_rate=a.rate)
