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

    def train_cnn(self):
        print("Training Starting")
        start_time = time.time()
        history = defaultdict(list)

        for t in range(self.epochs):
            correct = 0
            avg_valid_loss, valid_correct = 0, 0
            self.model.train()
            for batch, (X, y) in enumerate(self.train_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                self, self.optimizer.zero_grad()
                # Compute prediction and loss

                pred = self.model(X)

                loss = self.loss_fn(pred, y)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                # Store loss history for future plotting

                # Backpropagation
                loss.backward()
                self.optimizer.step()

            history["avg_train_loss"].append(loss.item())
            avg_train_loss = loss / self.train_num_batches
            accuracy = correct / self.train_size * 100
            history["train_accuracy"].append(accuracy)

            if self.verbose == True:
                print(f"Epoch {t+1} of {self.epochs}")
                print("-" * 15)
                print(
                    f"Training Results, Epoch {t+1}:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_train_loss.item():>8f} \n"
                )

                ###################### VALIDATION LOOP ##############################
            self.model.eval()
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
                print(
                    f"Validation Results, Epoch {t+1}: \n Accuracy: {(valid_accuracy):>0.1f}%, Avg loss: {avg_valid_loss:>8f} \n"
                )

        print("Done!")
        print(
            f"Final Train Accuracy: {(accuracy):>0.1f}%, and Avg loss: {avg_train_loss.item():>8f} \n"
        )
        print(
            f"Final Validation Accuracy: {(valid_accuracy):>0.1f}%, and Avg loss: {avg_valid_loss:>8f} \n"
        )
        current_time = time.time()
        total = current_time - start_time
        print(f"Training time: {round(total/60,2)} minutes")
        return history

    def train_spiking_model(self):
        start_time = time.time()
        print("Starting Training")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[50], gamma=0.5
        )

        for t in range(self.epochs):

            avg_train_loss = 0
            correct = 0
            avg_valid_loss, valid_correct = 0, 0

            self.model.train()
            for batch, (data_it, targets_it) in enumerate(self.train_loader):
                data_it = data_it.to(self.device)
                targets_it = targets_it.to(self.device)

                self.optimizer.zero_grad()

                # Compute prediction and loss
                spike_data = spikegen.rate(
                    data_it, num_steps=self.num_steps, gain=1, offset=0
                )  #! num_steps not defined
                spk_targets_it = torch.clamp(
                    spikegen.to_one_hot(targets_it, 10) * 1.05, min=0.05
                )

                spk_rec, mem_rec = self.model(
                    spike_data.view(self.num_steps, self.batch_size, -1)
                )

                # Sum loss over time steps: BPTT
                loss = torch.zeros(
                    (1), dtype=self.dtype, device=self.device  #! dtype not defined
                )  # creates a 1D tensor to store total loss over time.
                for step in range(self.num_steps):
                    loss += self.loss_fn(
                        mem_rec[step], spk_targets_it
                    )  # Loss at each time step is added to give total loss.

                avg_train_loss += loss

                _, predicted = spk_rec.sum(dim=0).max(1)
                correct += (predicted == targets_it).type(torch.float).sum().item()

                # Backpropagation
                loss.backward()
                self.optimizer.step()

            avg_train_loss /= self.train_num_batches
            accuracy = correct / self.train_size * 100
            self.history["avg_train_loss"].append(avg_train_loss.item())
            self.history["train_accuracy"].append(accuracy)

            if self.verbose == True:
                print(f"Epoch {t+1} of {self.epochs}")
                print("-" * 15)
                print(
                    f"Training Results, Epoch {t+1}:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_train_loss.item():>8f} \n"
                )

                ###################### VALIDATION LOOP ##############################
            self.model.eval()
            with torch.no_grad():
                for valid_data_it, valid_targets_it in self.valid_loader:
                    valid_data_it = valid_data_it.to(self.device)
                    valid_targets_it = valid_targets_it.to(self.device)

                    valid_spike_data = spikegen.rate(
                        valid_data_it, num_steps=self.num_steps, gain=1, offset=0
                    )
                    valid_spk_targets_it = torch.clamp(
                        spikegen.to_one_hot(targets_it, 10) * 1.05, min=0.05
                    )

                    valid_spk_rec, valid_mem_rec = self.model(
                        valid_spike_data.view(self.num_steps, self.batch_size, -1)
                    )

                    valid_loss = torch.zeros((1), dtype=self.dtype, device=self.device)
                    for step in range(self.num_steps):
                        valid_loss += self.loss_fn(
                            valid_mem_rec[step], valid_spk_targets_it
                        )

                    avg_valid_loss += valid_loss

                    _, valid_predicted = valid_spk_rec.sum(dim=0).max(1)
                    valid_correct += (
                        (valid_predicted == valid_targets_it)
                        .type(torch.float)
                        .sum()
                        .item()
                    )

            avg_valid_loss /= self.valid_num_batches
            valid_accuracy = valid_correct / self.valid_size * 100

            self.history["avg_valid_loss"].append(avg_valid_loss.item())
            self.history["valid_accuracy"].append(valid_accuracy)
            scheduler.step()

            if self.verbose == True:
                print(
                    f"Validation Results, Epoch {t+1}: \n Accuracy: {(valid_accuracy):>0.1f}%, Avg loss: {avg_valid_loss.item():>8f} \n"
                )

    def train_spiking_cnn(self):
        start_time = time.time()
        print('Starting Training')


        for t in range(self.epochs):
            
            avg_train_loss = 0
            correct = 0
            avg_test_loss, test_correct = 0, 0
            self.model.train()
            for batch, (data_it, targets_it) in enumerate(self.train_loader):
                data_it = data_it.to(self.device)
                targets_it = targets_it.to(self.device)

                self.optimizer.zero_grad()
                
                # Compute prediction and loss
                spike_data = spikegen.rate(data_it, num_steps=self.num_steps, gain=1, offset=0)
                spk_targets_it = torch.clamp(spikegen.to_one_hot(targets_it, 10) * 1.05, min=0.05)
                
                spk_rec, mem_rec = self.model(spike_data.view(self.num_steps,self.batch_size, 1,self.res,self.res)) 
                
                # Sum loss over time steps: BPTT
                loss = torch.zeros((1), dtype=self.dtype, device=self.device)   # creates a 1D tensor to store total loss over time. 
                for step in range(self.num_steps):
                    loss += self.loss_fn(mem_rec[step], spk_targets_it) # Loss at each time step is added to give total loss.

                avg_train_loss += loss
                
                _, predicted = spk_rec.sum(dim=0).max(1) 
                correct += (predicted == targets_it).type(torch.float).sum().item()

                # Backpropagation
                loss.backward()
                self.optimizer.step()

            avg_train_loss /= self.train_num_batches
            accuracy = correct / self.train_size * 100      
            self.history['avg_train_loss'].append(avg_train_loss.item())
            self.history['train_accuracy'].append(accuracy)
            
            if self.verbose == True: 
                print(f"Epoch {t+1} of {self.epochs}")
                print('-' * 15)
                print(f"Training Results, Epoch {t+1}:\n Accuracy: {(accuracy):>0.1f}%, Avg loss: {avg_train_loss.item():>8f} \n")

                ###################### TEST LOOP ##############################
            self.model.eval()
            with torch.no_grad():
                for test_data_it, test_targets_it in self.valid_loader:
                    test_data_it = test_data_it.to(self.device)
                    test_targets_it = test_targets_it.to(self.device)
                    
                    test_spike_data = spikegen.rate(test_data_it, num_steps=self.num_steps, gain=1, offset=0)
                    test_spk_targets_it = torch.clamp(spikegen.to_one_hot(targets_it, 10) * 1.05, min=0.05)

                    test_spk_rec, test_mem_rec = self.model(test_spike_data.view(self.num_steps,self.batch_size, 1, self.resolution, self.resolution)) 
                    
                    test_loss = torch.zeros((1),dtype = self.dtype, device = self.device)    
                    for step in range(self.num_steps):
                        test_loss += self.loss_fn(test_mem_rec[step], test_spk_targets_it)
                    
                    avg_test_loss += test_loss
                    
                    
                    _, test_predicted = test_spk_rec.sum(dim=0).max(1)
                    test_correct += (test_predicted == test_targets_it).type(torch.float).sum().item()
            

            avg_test_loss /= self.test_num_batches
            test_accuracy = test_correct / self.test_size * 100
                
            self.history['avg_valid_loss'].append(avg_test_loss.item())
            self.history['vtest_accuracy'].append(test_accuracy)
            
            if self.verbose == True: 
                print(f"Validation Results, Epoch {t+1}: \n Accuracy: {(self.valid_accuracy):>0.1f}%, Avg loss: {self.avg_valid_loss.item():>8f} \n")


        print("Done!")
        print(f"Final Train Accuracy: {(accuracy):>0.1f}%, and Avg loss: {avg_train_loss.item():>8f} \n")
        print(f"Final Validation Accuracy: {(self.valid_accuracy):>0.1f}%, and Avg loss: {self.avg_valid_loss.item():>8f} \n")
        current_time = time.time()
        total = current_time - start_time
        print(f'Training time: {round(total/60,2)} minutes')
        return self.history


    def _training_loop(self) -> Tuple[float,float]:
        correct: float = 0.0
        avg_train_loss: float = 0.0

        self.model.train()
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

        return (avg_train_loss, correct)

    def _spiking_training_loop(self):

        for batch, (data_it, targets_it) in enumerate(self.train_loader):
            data_it = data_it.to(self.device)
            targets_it = targets_it.to(self.device)

            self.optimizer.zero_grad()
            
            spike_data, spk_targets_it = self._rate_encoding(data_it, targets_it)
            
            spk_rec, mem_rec = self.model(spike_data)
            
            # Sum loss over time steps: BPTT
            loss = torch.zeros((1), dtype=self.dtype, device=self.device)   # creates a 1D tensor to store total loss over time. 
            for step in range(self.num_steps):
                loss += self.loss_fn(mem_rec[step], spk_targets_it) # Loss at each time step is added to give total loss.

            avg_train_loss += loss
            
            _, predicted = spk_rec.sum(dim=0).max(1) 
            correct += (predicted == targets_it).type(torch.float).sum().item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

    
    def _rate_encoding(self, data, targets):

        # convert data to rate encoded spikes.
        spike_data = spikegen.rate(data, num_steps=self.num_steps, gain=1, offset=0)
        spk_targets = torch.clamp(spikegen.to_one_hot(targets, 10) * 1.05, min=0.05)

        spike_data_view = spike_data.view(self.num_steps,self.batch_size, 1,self.res,self.res)

        return spike_data_view, spk_targets