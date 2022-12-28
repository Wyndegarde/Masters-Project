import torch
import torch.nn as nn
import snntorch as snn

from config import ModelParameters, TrainingParameters

# Define Network
class SnnNet(nn.Module):
    def __init__(
        self,
        res: int = ModelParameters.RESOLUTION,
        spike_grad: float = ModelParameters.SPIKE_GRADIENT,
        num_hidden: int = ModelParameters.NUM_HIDDEN,
        num_outputs: int = ModelParameters.NUM_OUTPUTS,
        beta: float = ModelParameters.BETA,
        batch_size: int = TrainingParameters.BATCH_SIZE,
        num_steps: int = ModelParameters.NUM_STEPS,
    ):
        super().__init__()

        self.res = res
        self.spike_grad = spike_grad
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.beta = beta
        self.batch_size = batch_size
        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = nn.Linear(self.res * self.res, self.num_hidden)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.fc3 = nn.Linear(self.num_hidden, self.num_outputs)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

    def forward(self, x):
        # Initialize hidden states + output spike at t=0
        spk1, mem1 = self.lif1.init_leaky(self.batch_size, self.num_hidden)
        spk2, mem2 = self.lif2.init_leaky(self.batch_size, self.num_hidden)
        spk3, mem3 = self.lif3.init_leaky(self.batch_size, self.num_outputs)

        spk3_rec = []
        mem3_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
