from typing import Any
from loguru import logger as log
from dataclasses import asdict

import inquirer
import numpy as np
import torch
from pymongo.errors import ServerSelectionTimeoutError

from interfaces import TrainingHistory
from handlers import DataHandler, DatabaseHandler, ModelHandler, TrainingHandler
from config import Config, LoguruSettings


log.add(
    LoguruSettings.MAIN_LOG,
    format=LoguruSettings.format,
    rotation=LoguruSettings.ROTATION,
    enqueue=True,
)


def main(model_type: str = "ANN"):

    # Set the seed for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    log.info(f"Using {Config.DEVICE} as device")

    data_handler: DataHandler = DataHandler()

    train_data, validation_data = data_handler.load_in_data()
    log.success("Data loaded in successfully")

    log.info("Getting model configurations")
    model_config = ModelHandler(model_type)

    training_handler: TrainingHandler = TrainingHandler(
        model_type,
        train_data,
        validation_data,
        model=model_config.model,
        loss_fn=model_config.loss_fn,
        spiking_model=model_config.spiking_model,
    )

    log.info(f"Beginning training of {model_type} model")
    history: TrainingHistory = training_handler.train_model()
    log.success(f"{model_type} model trained successfully")

    # Convert the dataclass to a dictionary for saving to MongoDB
    history_dict = asdict(history)

    # Requires MongoDB container to be running. If it's not, the history will not be saved
    try:
        log.info("Connecting to MongoDB")
        client = DatabaseHandler()
        client.insert_history(history_dict)
    except ServerSelectionTimeoutError:
        log.error("Could not connect to database. History not saved")
    finally:
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
