from typing import Any, Tuple
from collections import defaultdict
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from snntorch import surrogate
from snntorch import spikegen
import snntorch.spikeplot as splt

from services import TrainingServices
from config import Config, TrainingParameters, ModelParameters


class TrainingHandler:
    def __init__(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model: nn.Module,
        device: str = Config.DEVICE,
        epochs: int = TrainingParameters.EPOCHS,
        loss_fn: Any = TrainingParameters.SPIKING_LOSS_FUNCTION,
        verbose: bool = True,
        spiking_model: bool = False,
    ) -> None:

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.model = model.to(self.device)
        self.epochs = epochs
        self.loss_fn = loss_fn
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
        self.num_steps = ModelParameters.NUM_STEPS

        # ? Ingoring the types here as the len method for Datasets are weird.
        # Training variables.
        self.train_size = len(self.train_loader.dataset)  # type: ignore
        self.train_num_batches = len(self.train_loader)

        # validation variables
        self.valid_size = len(self.valid_loader.dataset)  # type: ignore
        self.valid_num_batches = len(self.valid_loader)

        # History dictionary to store the training metrics
        self.history: defaultdict = defaultdict(list)

    def train_model(self) -> defaultdict:

        start_time = time.time()
        print("Starting Training")

        for epoch in range(self.epochs):

            avg_train_loss, correct = self._training_loop()

            avg_valid_loss, valid_correct = self._validation_loop()

            self.add_metrics_to_history(
                avg_train_loss, correct, avg_valid_loss, valid_correct
            )
            # Calculate average training loss and accuracy
            avg_train_loss /= self.train_num_batches
            accuracy = correct / self.train_size * 100

            # Calculate average validation loss and accuracy
            avg_valid_loss /= self.valid_num_batches
            valid_accuracy = valid_correct / self.valid_size * 100

            # Store metric history for future plotting after going through all batches
            self.history["avg_train_loss"].append(avg_train_loss)
            self.history["train_accuracy"].append(accuracy)
            self.history["avg_valid_loss"].append(avg_valid_loss)
            self.history["valid_accuracy"].append(valid_accuracy)

            self.scheduler.step()

            if self.verbose == True:
                TrainingServices.print_epoch_metrics(
                    epoch,
                    self.epochs,
                    accuracy,
                    avg_train_loss,
                    avg_valid_loss,
                    valid_accuracy,
                )

        end_time = time.time()
        TrainingServices.print_final_metrics(
            accuracy, avg_train_loss, valid_accuracy, avg_valid_loss
        )
        return self.history

    def _training_loop(self) -> Tuple[float, float]:
        correct: float = 0.0
        avg_train_loss: float = 0.0

        self.model.train()
        for _, (train_data, train_labels) in enumerate(self.train_loader):
            # Zero out the gradients before passing to model
            self.optimizer.zero_grad()

            train_data = train_data.to(self.device)
            train_labels = train_labels.to(self.device)

            if self.spiking_model:

                predicted, loss = self._spiking_training_loop(train_data, train_labels)

            else:
                # Get model output
                predicted = self.model(train_data)

                # Calculate metrics for each batch
                loss = self.loss_fn(predicted, train_labels)

            # Add to metrics # ! issue here when running SNN
            correct += (
                (predicted.argmax(1) == train_labels).type(torch.float).sum().item()
            )
            avg_train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

        return (avg_train_loss, correct)

    def _spiking_training_loop(self, train_data, train_labels) -> Tuple[Tensor, Tensor]:
        spike_data, spk_targets_it = self._rate_encoding(train_data, train_labels)

        spk_rec, mem_rec = self.model(spike_data)

        # Sum loss over time steps: BPTT
        loss: Tensor = torch.zeros(
            (1), dtype=self.dtype, device=self.device
        )  # creates a 1D tensor to store total loss over time.
        for step in range(self.num_steps):
            loss += self.loss_fn(
                mem_rec[step], spk_targets_it
            )  # Loss at each time step is added to give total loss.

        _, predicted = spk_rec.sum(dim=0).max(1)

        return (predicted, loss)

    def _validation_loop(self) -> Tuple[float, float]:
        valid_correct: float = 0.0
        avg_valid_loss: float = 0.0

        self.model.eval()
        with torch.no_grad():
            for valid_data, valid_labels in self.valid_loader:

                valid_data = valid_data.to(self.device)
                valid_labels = valid_labels.to(self.device)

                if self.spiking_model:
                    valid_predicted, valid_loss = self._spiking_training_loop(
                        valid_data, valid_labels
                    )
                else:
                    # Get model output
                    valid_predicted = self.model(valid_data)

                    # Calculate metrics for each batch
                    valid_loss = self.loss_fn(valid_predicted, valid_labels)

                # Calculate average loss
                avg_valid_loss += valid_loss.item()
                valid_correct += (
                    (valid_predicted.argmax(1) == valid_labels)
                    .type(torch.float)
                    .sum()
                    .item()
                )
            return (avg_valid_loss, valid_correct)

    def _rate_encoding(self, data, targets):

        # convert data to rate encoded spikes.
        spike_data = spikegen.rate(data, num_steps=self.num_steps, gain=1, offset=0)
        spk_targets = torch.clamp(spikegen.to_one_hot(targets, 10) * 1.05, min=0.05)

        # # ! Could be differences between how to use view method for snn and scnn.
        # spike_data_view = spike_data.view(
        #     self.num_steps, self.batch_size, 1, self.res, self.res
        # )

        return spike_data, spk_targets

    def add_metrics_to_history(self, avg_train_loss, correct, avg_valid_loss, valid_correct):


        self.history["avg_train_loss"].append(avg_train_loss)
        self.history["train_accuracy"].append(correct / self.train_size * 100)
        self.history["avg_valid_loss"].append(avg_valid_loss)
        self.history["valid_accuracy"].append(valid_correct / self.valid_size * 100)