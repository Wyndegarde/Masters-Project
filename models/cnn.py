from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a different network
class CnnNet(nn.Module):
    def __init__(
        self,
        conv_kernel_size: int,
        conv_padding_size: int,
        mp_kernel_size: int,
        mp_stride_length: int,
        dropout: float,
        num_hidden: int,
        num_outputs: int,
        output_sizes: List[int],
    ) -> None:
        super().__init__()

        self.convolutions = nn.Sequential(
            # Do I change channels to a variable incase I end up with RGB images? ## Padding = 0 as all information is at the centre of image (may change if lower resolution)
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride_length),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),  #'same'
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride_length),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=conv_kernel_size,
                padding=conv_padding_size,
            ),
            nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride_length),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * output_sizes[-1] * output_sizes[-1], num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_outputs),
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)
