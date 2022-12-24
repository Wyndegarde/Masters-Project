from typing import Tuple, Any, Union, Sized

import torch
from torchvision import datasets, transforms
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

from config import Config, TrainingParameters, ModelParameters, PathSettings


class DataHandler:
    def __init__(
        self,
        data_path: str = PathSettings.DATASET_PATH,
        res: int = ModelParameters.RESOLUTION,
        batch_size: int = TrainingParameters.BATCH_SIZE,
        ratio: float = TrainingParameters.RATIO,
    ) -> None:

        self.data_path = data_path
        self.batch_size = batch_size
        self.res = res
        self.ratio = ratio

    def load_in_data(self) -> Tuple[DataLoader, DataLoader]:

        # change each image array to a tensor which automatically scales inputs to [0,1]
        transform = transforms.Compose(
            [
                transforms.Resize((self.res, self.res)),  # Resize images to 28*28
                transforms.Grayscale(),  # Make sure image is grayscale
                transforms.ToTensor(),
            ]
        )

        # Download training set and apply transformations.
        mnist_train: Dataset = datasets.MNIST(
            self.data_path, train=True, download=True, transform=transform
        )

        # same for test set
        mnist_test: Dataset = datasets.MNIST(
            self.data_path, train=False, download=True, transform=transform
        )

        # ? Ingoring the types here as the len method for Datasets are weird.
        train_len: int = int(len(mnist_train) / self.ratio)  # type: ignore
        dummy_len: int = len(mnist_train) - train_len  # type: ignore

        train_dataset, _ = random_split(
            mnist_train,
            (train_len, dummy_len),
            generator=torch.Generator().manual_seed(Config.SEED),
        )  # type: ignore

        # Load the data into the DataLoader so it's passed through the model, shuffled in batches.
        train_loader: DataLoader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        test_loader: DataLoader = DataLoader(
            mnist_test, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        return (train_loader, test_loader)
