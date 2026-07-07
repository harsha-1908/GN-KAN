import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Standard MLP Feed-Forward Block

    Input:
        (B, C, H, W)

    Output:
        (B, C, H, W)

    Components:
        1. Pointwise expansion
        2. GELU activation
        3. Pointwise projection
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = None,
        drop: float = 0.0,
    ):
        super().__init__()

        hidden_channels = hidden_channels or (channels * 4)

        self.channels = channels
        self.hidden_channels = hidden_channels

        self.drop = (
            nn.Dropout2d(drop)
            if drop > 0.0
            else nn.Identity()
        )

        # Expansion Projection
        self.fc1 = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=True
        )

        # Activation
        self.act = nn.GELU()

        # Projection
        self.fc2 = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=True
        )

        self._init_weights()

    def _init_weights(self):

        nn.init.kaiming_normal_(
            self.fc1.weight,
            mode="fan_out",
            nonlinearity="relu"
        )
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_normal_(
            self.fc2.weight,
            mode="fan_out",
            nonlinearity="relu"
        )
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):

        # 1. Expansion
        x = self.fc1(x)

        # 2. Activation
        x = self.act(x)

        x = self.drop(x)

        # 3. Projection
        x = self.fc2(x)

        x = self.drop(x)

        return x