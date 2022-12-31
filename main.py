from typing import Any
from loguru import logger as log

import torch
import torch.nn as nn
import numpy as np
import inquirer

from config import Config, LoguruSettings, TrainingParameters
from handlers import DataHandler, TrainingHandler
from models import AnnNet, CnnNet, SnnNet, ScnnNet

log.add(
    LoguruSettings.MAIN_LOG,
    format=LoguruSettings.format,
    rotation=LoguruSettings.ROTATION,
    enqueue=True,
)


def main(model_type: str = "ANN"):

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    log.info(f"Using {Config.DEVICE} as device")

    data_handler = DataHandler()

    train_data, validation_data = data_handler.load_in_data()

    match model_type:
        case "ANN":
            model: nn.Module = AnnNet()
        case "CNN":
            model = CnnNet()
        case "SNN":
            model = SnnNet()
        case "SCNN":
            model = ScnnNet()
        case _:
            raise ValueError("Model type not recognised")

    if model_type in ["SNN", "SCNN"]:
        log.info("Training a Spiking Nueral Network")

        # define the spiking model specific parameters
        spiking_model: bool = True
        loss_fn: Any = nn.MSELoss()
    else:
        log.info("Training a Deep Nueral Network")

        spiking_model = False
        loss_fn = TrainingParameters.LOSS_FUNCTION

    training_handler = TrainingHandler(
        train_data, validation_data, model, loss_fn=loss_fn, spiking_model=spiking_model
    )

    history = training_handler.train_model()

    # Plot the history

    print(history)

    return history


if __name__ == "__main__":

    questions = [
        inquirer.List(
            "model",
            message="Which model do you want to use?",
            choices=["ANN", "CNN", "SNN", "SCNN"],
        ),
    ]
    answers = inquirer.prompt(questions)

    main(answers["model"])
