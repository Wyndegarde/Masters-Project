from typing import Any
from loguru import logger as log
from dataclasses import asdict

import inquirer
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn

from interfaces import TrainingHistory
from models import AnnNet, CnnNet, SnnNet, ScnnNet
from handlers import DataHandler, DatabaseHandler, ModelHandler, TrainingHandler
from config import Config, LoguruSettings, TrainingParameters, ModelParameters


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

    data_handler: DataHandler = DataHandler()

    train_data, validation_data = data_handler.load_in_data()

    model_config = ModelHandler(model_type)

    training_handler: TrainingHandler = TrainingHandler(
        train_data,
        validation_data,
        model=model_config.model,
        loss_fn=model_config.loss_fn,
        spiking_model=model_config.spiking_model,
    )

    history: TrainingHistory = training_handler.train_model()

    history_dict = asdict(history)

    model_id = f"{model_type}_Res{ModelParameters.RESOLUTION}_Ratio{int(TrainingParameters.RATIO * 100)}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}".lower()

    history_dict["_id"] = model_id

    try:
        client = DatabaseHandler()
        client.insert_history(history_dict)
    except:
        log.error("Could not connect to database")
        raise

    # TODO: Plot the history
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
