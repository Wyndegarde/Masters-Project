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

    data_handler = DataHandler()

    train_data, validation_data = data_handler.load_in_data()

    match model_type:
        case "ANN":
            model: nn.Module = AnnNet()
        case "CNN":
            model: nn.Module = CnnNet()  # type: ignore
        case "SNN":
            model: nn.Module = SnnNet()  # type: ignore
        case "SCNN":
            model: nn.Module = ScnnNet()  # type: ignore
        case _:
            raise ValueError("Model type not recognised")

    training_handler = TrainingHandler(train_data, validation_data, model)

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
