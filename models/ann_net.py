import torch.nn as nn
import torch.nn.functional as F


from config import ModelParameters

# Define Network
class AnnNet(nn.Module):
    def __init__(
        self,
        res: int = ModelParameters.RESOLUTION,
        num_hidden: int = ModelParameters.NUM_HIDDEN,
        num_outputs: int = ModelParameters.NUM_OUTPUTS,
    ) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            res * res, num_hidden
        )  # input layer with as many neurons as pixels.
        self.fc2 = nn.Linear(
            num_hidden, num_hidden
        )  # Second Dense/linear layer that receives the output spikes from previous layer
        self.fc3 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.log_softmax(
            self.fc3(x), dim=1
        )  # dim = 1 sums the rows so they equal 1. I.e. each input.
