from typing import Dict
from torch import Tensor


class TrainingServices:
    @staticmethod
    def calculate_loss_metrics():
        pass

    @staticmethod
    def print_epoch_metrics(
        epoch: int,
        epochs: int,
        accuracy: float,
        avg_train_loss: float,
        avg_valid_loss: float,
        valid_accuracy: float,
    ) -> None:

        print(f"Epoch {epoch+1} of {epochs}")
        print("-" * 15)
        print(
            f"Training Results, Epoch {epoch+1}:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_train_loss:>8f} \n"
        )
        print("-" * 15)
        print(
            f"Validation Results, Epoch {epoch+1}: \n Accuracy: {(valid_accuracy):>0.1f}%, Avg loss: {avg_valid_loss:>8f} \n"
        )
