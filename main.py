import torch
import numpy as np

from config import Config
from handlers import DataHandler, TrainingHandler
from models import AnnNet


def main():

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    data_handler = DataHandler()

    train_data, validation_data = data_handler.load_in_data()

    model = AnnNet()

    training_handler = TrainingHandler(train_data, validation_data, model)

    history = training_handler.train_model()

    print(history)

    return history


if __name__ == "__main__":
    main()
