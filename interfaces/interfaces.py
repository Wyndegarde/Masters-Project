from typing import List

from dataclasses import dataclass


@dataclass
class TrainingHistory:
    best_epoch: int
    best_valid_accuracy: float
    train_accuracies: List[float]
    train_loss: List[float]
    valid_accuracies: List[float]
    valid_loss: List[float]
