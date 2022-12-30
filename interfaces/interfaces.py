from typing import List

from dataclasses import dataclass


@dataclass
class BatchMetrics:
    correct: float
    loss: float
    accuracy: float


@dataclass
class TrainingMetrics:
    train_correct: List[float]
    train_loss: List[float]
    valid_correct: List[float]
    valid_loss: List[float]


@dataclass
class EpochMetrics:
    train_loss: float
    train_accuracy: float
    valid_loss: float
    valid_accuracy: float


@dataclass
class TrainingHistory:
    metrics: List[TrainingMetrics]
