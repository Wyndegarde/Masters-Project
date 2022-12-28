import torch.nn as nn
import torch.nn.functional as F


from config import ModelParameters


class AnnNet(nn.Module):
    """This is a simple ANN with 2 hidden layers."""

    def __init__(
        self,
        res: int = ModelParameters.RESOLUTION,
        num_hidden: int = ModelParameters.NUM_HIDDEN,
        num_outputs: int = ModelParameters.NUM_OUTPUTS,
    ) -> None:
        super().__init__()

        # Set the model parameters
        self.res = res
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Flatten the input image to one dimension.
        self.flatten = nn.Flatten()

        # Create the sequence of fully connected layers
        self.fc1 = nn.Linear(self.res * self.res, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.fc3 = nn.Linear(self.num_hidden, self.num_outputs)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # dim = 1 sums the rows so they equal 1. I.e. each input.
        return F.log_softmax(self.fc3(x), dim=1)
