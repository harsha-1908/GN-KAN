import torch
import torch.nn as nn


class SD(nn.Module):

    def __init__(
        self,
        channels,
        hidden_channels=None,
        spatial_kernel=3
    ):
        super().__init__()

        hidden_channels = (
            hidden_channels
            or channels * 4
        )

        self.fc1 = nn.Conv2d(
            channels,
            hidden_channels,
            1
        )

        self.act = nn.GELU()

        pad = (spatial_kernel - 1) // 2

        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            spatial_kernel,
            padding=pad,
            groups=hidden_channels
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                1
            ),
            nn.Sigmoid()
        )

        self.fc2 = nn.Conv2d(
            hidden_channels,
            channels,
            1
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.act(x)

        x = self.depthwise(x)

        x = x * self.gate(x)

        x = self.fc2(x)

        return x