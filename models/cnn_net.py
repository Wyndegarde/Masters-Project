from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelParameters
from utils import Utilities

# Define a different network
class CnnNet(nn.Module):
    def __init__(
        self,
        res: int,
        conv_kernel_size: int = ModelParameters.CONV_KERNEL_SIZE,
        conv_padding_size: int = ModelParameters.CONV_PADDING_SIZE,
        mp_kernel_size: int = ModelParameters.MP_KERNEL_SIZE,
        mp_padding_size: int = ModelParameters.MP_PADDING_SIZE,
        mp_stride_length: int = ModelParameters.MP_STRIDE_LENGTH,
        dropout: float = ModelParameters.DROPOUT,
        num_hidden: int = ModelParameters.NUM_HIDDEN,
        num_outputs: int = ModelParameters.NUM_OUTPUTS,
    ) -> None:
        super().__init__()

        self.res = res
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding_size = conv_padding_size
        self.mp_kernel_size = mp_kernel_size
        self.mp_padding_size = mp_padding_size
        self.mp_stride_length = mp_stride_length
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.output_sizes = Utilities.get_output_sizes(
            self.res,
            self.conv_kernel_size,
            self.conv_padding_size,
            self.mp_kernel_size,
            self.mp_padding_size,
            self.mp_stride_length,
        )

        self.convolutions = nn.Sequential(
            # Do I change channels to a variable incase I end up with RGB images? ## Padding = 0 as all information is at the centre of image (may change if lower resolution)
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding_size,
            ),
            nn.MaxPool2d(kernel_size=self.mp_kernel_size, stride=self.mp_stride_length),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding_size,
            ),  #'same'
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding_size,
            ),
            nn.MaxPool2d(kernel_size=self.mp_kernel_size, stride=self.mp_stride_length),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding_size,
            ),
            nn.MaxPool2d(kernel_size=self.mp_kernel_size, stride=self.mp_stride_length),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(256 * self.output_sizes[-1] * self.output_sizes[-1], num_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_outputs),
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)
