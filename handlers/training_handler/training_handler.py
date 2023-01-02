from typing import Any, Tuple, List
from collections import defaultdict
import time
from loguru import logger as log

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from snntorch import surrogate
from snntorch import spikegen
import snntorch.spikeplot as splt

from services import TrainingServices
from interfaces import TrainingHistory
from config import Config, LoguruSettings, TrainingParameters, ModelParameters

log.add(
    LoguruSettings.TRAINING_HANDLER_LOG,
    format=LoguruSettings.format,
    rotation=LoguruSettings.ROTATION,
    enqueue=True,
)


class TrainingHandler:
    def __init__(
        self,
        model_type: str,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model: nn.Module,
        device: str = Config.DEVICE,
        epochs: int = TrainingParameters.EPOCHS,
        loss_fn: Any = TrainingParameters.LOSS_FUNCTION,
        num_steps: int = ModelParameters.NUM_STEPS,
        verbose: bool = True,
        spiking_model: bool = False,
    ) -> None:

        self.model_type = model_type
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.model = model.to(self.device)
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.verbose = verbose
        self.spiking_model = spiking_model

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=TrainingParameters.LEARNING_RATE,
            betas=TrainingParameters.BETAS,
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[50], gamma=0.5
        )

        self.dtype = torch.float

        # ? Ingoring the types here as the len method for Datasets are weird.
        # Training variables.
        self.train_size = len(self.train_loader.dataset)  # type: ignore
        self.train_num_batches = len(self.train_loader)

        # validation variables
        self.valid_size = len(self.valid_loader.dataset)  # type: ignore
        self.valid_num_batches = len(self.valid_loader)

    def train_model(self) -> TrainingHistory:
        """
        This method trains the selected model using the parameters defined in the config file.

        Returns:
            defaultdict: Contains the training and validation metrics for each epoch.
        """

        train_losses: List[float] = []
        train_accuracies: List[float] = []
        valid_losses: List[float] = []
        valid_accuracies: List[float] = []

        start_time = time.time()
        print("Starting Training")

        for epoch in range(self.epochs):

            avg_train_loss, correct = self._training_loop()

            avg_valid_loss, valid_correct = self._validation_loop()

            # Calculate average training loss and accuracy
            avg_train_loss /= self.train_num_batches
            accuracy = correct / self.train_size * 100

            # Calculate average validation loss and accuracy
            avg_valid_loss /= self.valid_num_batches
            valid_accuracy = valid_correct / self.valid_size * 100

            # Append epoch metrics to lists
            train_losses.append(avg_train_loss)
            train_accuracies.append(accuracy)
            valid_losses.append(avg_valid_loss)
            valid_accuracies.append(valid_accuracy)

            if self.verbose == True:
                TrainingServices.print_epoch_metrics(
                    epoch,
                    self.epochs,
                    accuracy,
                    avg_train_loss,
                    avg_valid_loss,
                    valid_accuracy,
                )

            self.scheduler.step()

        end_time = time.time()
        time_taken = (end_time - start_time) / 60

        log.success(f"Training completed in {time_taken} minutes")

        TrainingServices.print_final_metrics(
            accuracy, avg_train_loss, valid_accuracy, avg_valid_loss
        )

        # Get the best epoch and accuracy for the validation set.
        best_epoch, best_valid_accuracy = TrainingServices.get_best_epoch(
            valid_accuracies
        )

        # Create a unique id for the model to be inserted to the db.
        model_id = TrainingServices.create_model_id(self.model_type)

        # Store all relevant information for the training history.
        history = TrainingHistory(
            model_id,
            best_epoch,
            best_valid_accuracy,
            train_accuracies,
            train_losses,
            valid_accuracies,
            valid_losses,
        )

        return history

    def _training_loop(self) -> Tuple[float, float]:
        """
        Thiis method performs the training loop for a single epoch.

        Returns:
            Tuple[float, float]: Returns the average loss and number of correct predictions for the epoch.
        """
        correct: float = 0.0
        avg_train_loss: float = 0.0

        self.model.train()
        for _, (train_data, train_labels) in enumerate(self.train_loader):
            # Zero out the gradients before passing to model
            self.optimizer.zero_grad()

            # Move data to device (GPU if available)
            train_data = train_data.to(self.device)
            train_labels = train_labels.to(self.device)

            # alternative method of passing data to model if spiking model
            if self.spiking_model:

                # converts data to spikes and gets the output metrics
                predicted, loss = self._spiking_training_loop(train_data, train_labels)
                correct += (predicted == train_labels).type(torch.float).sum().item()

            # if not a spike model, pass data to model as normal
            else:
                # Get model output
                predicted = self.model(train_data)

                # Calculate metrics for each batch
                loss = self.loss_fn(predicted, train_labels)

                # Add to metrics
                correct += (
                    (predicted.argmax(1) == train_labels).type(torch.float).sum().item()
                )
            # Add loss to average loss (avg to be calculated later)
            avg_train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

        return (avg_train_loss, correct)

    def _spiking_training_loop(self, train_data, train_labels) -> Tuple[Tensor, Tensor]:
        """
        This method converts the data to spikes and passes it to the model for training.

        Args:
            train_data (DataLoader): Images used for training.
            train_labels (DataLoader): Labels associated with each of the images.

        Returns:
            Tuple[Tensor, Tensor]: The predicted labels and the loss for the batch.
        """

        # Use rate encoding to convert data to spikes
        spike_data, spk_targets_it = self._rate_encoding(train_data, train_labels)

        spk_rec, mem_rec = self.model(spike_data)

        # Sum loss over time steps: BPTT
        loss: Tensor = torch.zeros(
            (1), dtype=self.dtype, device=self.device
        )  # creates a 1D tensor to store total loss over time.

        for step in range(self.num_steps):
            # Loss at each time step is added to give total loss.
            loss += self.loss_fn(mem_rec[step], spk_targets_it)

        _, predicted = spk_rec.sum(dim=0).max(1)

        return (predicted, loss)

    def _validation_loop(self) -> Tuple[float, float]:
        """
        This method performs the validation loop for a single epoch.

        Returns:
            Tuple[float, float]: The average loss and number of correct predictions for the validation data for the epoch.
        """

        # Initialize metrics
        valid_correct: float = 0.0
        avg_valid_loss: float = 0.0

        # Set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for valid_data, valid_labels in self.valid_loader:

                valid_data = valid_data.to(self.device)
                valid_labels = valid_labels.to(self.device)

                # alternative method of passing data to model if spiking model
                if self.spiking_model:

                    # use the same method as training loop to pass data to spiking model
                    valid_predicted, valid_loss = self._spiking_training_loop(
                        valid_data, valid_labels
                    )
                    valid_correct += (
                        (valid_predicted == valid_labels).type(torch.float).sum().item()
                    )
                else:
                    # Get model output
                    valid_predicted = self.model(valid_data)

                    # Calculate metrics for each batch
                    valid_loss = self.loss_fn(valid_predicted, valid_labels)
                    valid_correct += (
                        (valid_predicted.argmax(1) == valid_labels)
                        .type(torch.float)
                        .sum()
                        .item()
                    )

                # Calculate average loss
                avg_valid_loss += valid_loss.item()

            return (avg_valid_loss, valid_correct)

    def _rate_encoding(self, data, targets):

        # convert data to rate encoded spikes.
        spike_data = spikegen.rate(data, num_steps=self.num_steps, gain=1, offset=0)
        # Convert targets to one-hot encoded spikes
        spk_targets = torch.clamp(spikegen.to_one_hot(targets, 10) * 1.05, min=0.05)

        return spike_data, spk_targets
