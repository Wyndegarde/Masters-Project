import os
from typing import Tuple
import logging

import numpy as np
import torch
import torch.nn as nn

from snntorch import surrogate


class Config:

    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class LoguruSettings:
    """Class to hold Loguru settings"""

    # Loguru parameters
    LEVEL: int = logging.DEBUG
    ROTATION: str = "1 day"
    format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "file: <cyan>{file}</cyan> -> path: <cyan>{name}</cyan> -> function: <cyan>{function}</cyan> -> line: <cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Loguru filepaths
    MAIN_LOG: str = "logs/main.log"
    TRAINING_HANDLER_LOG: str = "logs/training_handler.log"


class ModelParameters:
    """Class to hold model parameters"""

    # Shared Parameters
    RESOLUTION: int = 28
    NUM_HIDDEN: int = 128
    NUM_OUTPUTS: int = 10

    # CNN Parameters
    CONV_KERNEL_SIZE: int = 3
    CONV_PADDING_SIZE: int = 1
    CONV_STRIDE: int = 1
    MP_KERNEL_SIZE: int = 2
    MP_PADDING_SIZE: int = 0
    MP_STRIDE_LENGTH: int = 2
    DROPOUT: float = 0.25

    # SNN parameters
    SLOPE: int = 5
    SPIKE_GRADIENT: surrogate = surrogate.fast_sigmoid(slope = SLOPE)
    NUM_STEPS: int = 10
    TIME_STEP: float = 1e-3
    TAU_MEM: float = 2e-2
    BETA: float = float(np.exp(-TIME_STEP / TAU_MEM))


class TrainingParameters:
    """Class to hold training parameters"""

    RATIO: float = 1
    BATCH_SIZE: int = 64
    EPOCHS: int = 3
    LEARNING_RATE: float = 5e-4
    BETAS: Tuple[float, float] = (0.9, 0.999)
    LOSS_FUNCTION: nn.NLLLoss = nn.NLLLoss()


class PathSettings:
    """Class to hold filepaths"""

    DATASET_PATH: str = os.path.join(
        os.path.dirname(os.path.realpath("__file__")), "data", "datasets"
    )
