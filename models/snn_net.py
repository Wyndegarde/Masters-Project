import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
import snntorch.spikeplot as splt


# Define Network
class SnnNet(nn.Module):
    def __init__(
        self,
        res: int,
        spike_grad: float,
        num_hidden: int,
        num_outputs: int,
        beta: float,
        batch_size: int,
        num_steps: int,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Initialize layers
        self.fc1 = nn.Linear(res * res, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

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
