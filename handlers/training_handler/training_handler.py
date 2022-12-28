from typing import Any
from collections import defaultdict
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from snntorch import surrogate
from snntorch import spikegen
import snntorch.spikeplot as splt

from interfaces import EpochMetrics, TrainingHistory, TrainingMetrics
from services import TrainingServices
from config import Config, TrainingParameters


class TrainingHandler:
    def __init__(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model: nn.Module,
        device: str = Config.DEVICE,
        epochs: int = TrainingParameters.EPOCHS,
        loss_fn: Any = TrainingParameters.LOSS_FUNCTION,
        verbose: bool = True,
    ) -> None:

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.model = model.to(self.device)
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.verbose = verbose

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=TrainingParameters.LEARNING_RATE,
            betas=TrainingParameters.BETAS,
        )

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

            # Instantiate metrics for each epoch
            correct: float = 0.0
            avg_train_loss: float = 0.0
            avg_valid_loss: float = 0.0
            valid_correct: float = 0.0

            # Iterate through the training set by batches
            for _, (train_data, train_labels) in enumerate(self.train_loader):
                # Zero out the gradients before passing to model
                self.optimizer.zero_grad()

                train_data = train_data.to(self.device)
                train_labels = train_labels.to(self.device)

                # Get model output
                pred = self.model(train_data)

                # Calculate metrics for each batch
                loss: Tensor = self.loss_fn(pred, train_labels)

                correct += (
                    (pred.argmax(1) == train_labels).type(torch.float).sum().item()
                )

                avg_train_loss += loss.item()

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                ###################### VALIDATION LOOP ##############################
            with torch.no_grad():
                for valid_X, valid_y in self.valid_loader:

                    valid_X = valid_X.to(self.device)
                    valid_y = valid_y.to(self.device)

                    # Get model output
                    valid_pred = self.model(valid_X)

                    # Calculate metrics for each batch
                    valid_loss: Tensor = self.loss_fn(valid_pred, valid_y)

                    # Calculate average loss
                    avg_valid_loss += valid_loss.item()
                    valid_correct += (
                        (valid_pred.argmax(1) == valid_y).type(torch.float).sum().item()
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

            if self.verbose == True:
                TrainingServices.print_epoch_metrics(
                    epoch,
                    self.epochs,
                    accuracy,
                    avg_train_loss,
                    avg_valid_loss,
                    valid_accuracy,
                )

        print("Done!")
        print(
            f"Final Train Accuracy: {(accuracy):>0.1f}%, and Avg loss: {avg_train_loss:>8f} \n"
        )
        print(
            f"Final Validation Accuracy: {(valid_accuracy):>0.1f}%, and Avg loss: {avg_valid_loss:>8f} \n"
        )
        return self.history

    def _process_batch(
        self,
        data: Tensor,
        targets: Tensor,
        epoch_metrics: EpochMetrics,
        training_loop: bool = True,
    ) -> Tensor:

        data = data.to(self.device)
        targets = targets.to(self.device)

        # Get model output
        pred = self.model(data)

        # Calculate metrics for each batch
        loss: Tensor = self.loss_fn(pred, targets)

        # print(correct)
        # correct += (pred.argmax(1) == targets).type(torch.float).sum().item()
        # print(correct)
        # avg_loss += loss.item()

        return loss

    def train_spiking_model(self):
        pass

    def prep_spiking_data(self, data, targets, model):

        # Compute prediction and loss
        spike_data = spikegen.rate(data, num_steps=self.num_steps, gain=1, offset=0)
        spk_targets_it = torch.clamp(spikegen.to_one_hot(targets, 10) * 1.05, min=0.05)

        # spk_rec, mem_rec = model(
        #     spike_data.view(num_steps, batch_size, 1, self.res, self.res)
        # )  # !

    # def get_ann_results(
    #     self,
    #     resolution,
    #     epochs=20,
    #     slope=25,
    #     loss_upper=1.05,
    #     acc_lower=0,
    #     acc_higher=100,
    #     verbose=True,
    # ):
    #     train, valid, test = self.load_in_data(resolution)
    #     model = Ann_Net(resolution).to(device)

    #     output = train_model(train, valid, model, epochs, verbose=verbose)
    #     plot_training_history(
    #         output,
    #         resolution,
    #         ylimita=loss_upper,
    #         ylimitb_lower=acc_lower,
    #         ylimitb_upper=acc_higher,
    #     )

    #     return output
