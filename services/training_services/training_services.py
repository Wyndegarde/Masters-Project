from typing import Tuple, List


class TrainingServices:
    @staticmethod
    def get_best_epoch(valid_accuracy: List[float]) -> Tuple[int, float]:
        """
        This method finds the epoch with the highest accuracy.

        Args:
            valid_accuracy (List[float]): List of validation accuracies.

        Returns:
            Tuple[int, float]: The epoch with the highest accuracy and the accuracy.
        """
        best_idx = valid_accuracy.index(max(valid_accuracy))

        best_epoch = best_idx + 1
        best_valid_accuracy = valid_accuracy[best_idx]

        return (best_epoch, best_valid_accuracy)

    @staticmethod
    def create_id():
        """
        This method creates a unique id for the training history using the model type, resolution, and training size

        Returns:
            str: Unique id for the training history
        """
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

    @staticmethod
    def print_final_metrics(
        accuracy: float,
        avg_train_loss: float,
        valid_accuracy: float,
        avg_valid_loss: float,
    ) -> None:

        print(
            f"Training Results:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_train_loss:>8f} \n"
        )
        print("-" * 15)
        print(
            f"Validation Results: \n Accuracy: {(valid_accuracy):>0.1f}%, Avg loss: {avg_valid_loss:>8f} \n"
        )
