from loguru import logger as log

import torch
import torch.nn as nn
import numpy as np
import inquirer

from config import Config, LoguruSettings
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

    spiking_model: bool = model_type in ["SNN", "SCNN"]

    training_handler = TrainingHandler(train_data, validation_data, model, spiking_model=spiking_model)

    history = training_handler.train_model()

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
