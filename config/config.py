import os
from typing import Tuple

import torch
import torch.nn as nn


class Config:

    SEED: int = 42

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModelParameters:

    RESOLUTION: int = 28
    NUM_HIDDEN: int = 128
    NUM_OUTPUTS: int = 10


class TrainingParameters:

    RATIO: float = 1
    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    LEARNING_RATE: float = 5e-4
    BETAS: Tuple[float, float] = (0.9, 0.999)
    LOSS_FUNCTION: nn.NLLLoss = nn.NLLLoss()
    MOMENTUM: float = 0.9


class PathSettings:

    DATASET_PATH: str = os.path.join(
        os.path.dirname(os.path.realpath("__file__")), "data", "datasets"
    )
