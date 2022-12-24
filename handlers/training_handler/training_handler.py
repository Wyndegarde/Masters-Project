from typing import Any
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, TrainingParameters


class TrainingHandler:
    def __init__(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model: nn.Module,
        epochs: int = TrainingParameters.EPOCHS,
        loss_fn: Any = TrainingParameters.LOSS_FUNCTION,
        device: str = Config.DEVICE,
        verbose: bool = True,
    ) -> None:

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.device = device
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
        self.num_batches = len(self.valid_loader)

    def train_model(self) -> defaultdict:

        # Create a dictionary to store the output metrics
        history: defaultdict = defaultdict(list)

        for epoch in range(self.epochs):

            # Instantiate metrics for each epoch
            correct: float = 0
            avg_valid_loss: float = 0
            valid_correct: int = 0

            # Iterate through the training set by batches
            for _, (X, y) in enumerate(self.train_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                # Zero out the gradients before passing to model
                self.optimizer.zero_grad()

                # Compute prediction and loss
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Backpropagation
                loss.backward()
                self.optimizer.step()

            # Store loss history for future plotting after going through all batches
            history["avg_train_loss"].append(loss.item())
            avg_train_loss = loss / self.train_num_batches
            accuracy = correct / self.train_size * 100
            history["train_accuracy"].append(accuracy)

            if self.verbose == True:
                print(f"Epoch {epoch+1} of {self.epochs}")
                print("-" * 15)
                print(
                    f"Training Results, Epoch {epoch+1}:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_train_loss.item():>8f} \n"
                )

                ###################### VALIDATION LOOP ##############################
            with torch.no_grad():
                for valid_X, valid_y in self.valid_loader:
                    valid_X = valid_X.to(self.device)
                    valid_y = valid_y.to(self.device)

                    valid_pred = self.model(valid_X)
                    valid_loss = self.loss_fn(valid_pred, valid_y).item()
                    avg_valid_loss += self.loss_fn(valid_pred, valid_y).item()
                    valid_correct += (
                        (valid_pred.argmax(1) == valid_y).type(torch.float).sum().item()
                    )

            avg_valid_loss /= self.num_batches
            valid_accuracy = valid_correct / self.valid_size * 100

            history["avg_valid_loss"].append(avg_valid_loss)
            history["valid_accuracy"].append(valid_accuracy)

            if self.verbose == True:
                print(f"Epoch {epoch+1} of {self.epochs}")
                print("-" * 15)
                print(
                    f"Validation Results, Epoch {epoch+1}: \n Accuracy: {(valid_accuracy):>0.1f}%, Avg loss: {avg_valid_loss:>8f} \n"
                )

        print("Done!")
        print(
            f"Final Train Accuracy: {(accuracy):>0.1f}%, and Avg loss: {avg_train_loss.item():>8f} \n"
        )
        print(
            f"Final Validation Accuracy: {(valid_accuracy):>0.1f}%, and Avg loss: {avg_valid_loss:>8f} \n"
        )
        return history

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
