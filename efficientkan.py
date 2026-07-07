import torch
import torch.nn as nn


class BSplineActivation(nn.Module):
    """
    Lightweight EfficientKAN-style activation.

    Learnable piecewise basis approximation.
    """

    def __init__(
        self,
        channels,
        num_basis=8
    ):
        super().__init__()

        self.channels = channels
        self.num_basis = num_basis

        self.coeffs = nn.Parameter(
            torch.randn(
                1,
                channels,
                num_basis,
                1,
                1
            ) * 0.02
        )

        grid = torch.linspace(
            -2.0,
            2.0,
            num_basis
        )

        self.register_buffer(
            "grid",
            grid
        )

    def forward(self, x):

        basis_outputs = []

        for i in range(self.num_basis):

            basis = torch.exp(
                -(
                    x - self.grid[i]
                ) ** 2
            )

            basis_outputs.append(
                basis
            )

        basis_outputs = torch.stack(
            basis_outputs,
            dim=2
        )

        out = (
            basis_outputs
            * self.coeffs
        ).sum(dim=2)

        return out


class EfficientKAN(nn.Module):
    """
    EfficientKAN FFN Block

    Input:
        (B,C,H,W)

    Output:
        (B,C,H,W)
    """

    def __init__(
        self,
        channels,
        hidden_channels=None,
        drop=0.0,
        num_basis=8
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

        self.kan_act = BSplineActivation(
            hidden_channels,
            num_basis=num_basis
        )

        self.fc2 = nn.Conv2d(
            hidden_channels,
            channels,
            1
        )

        self.drop = (
            nn.Dropout2d(drop)
            if drop > 0
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):

        nn.init.kaiming_normal_(
            self.fc1.weight,
            mode="fan_out"
        )

        nn.init.zeros_(
            self.fc1.bias
        )

        nn.init.kaiming_normal_(
            self.fc2.weight,
            mode="fan_out"
        )

        nn.init.zeros_(
            self.fc2.bias
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.kan_act(x)

        x = self.drop(x)

        x = self.fc2(x)

        x = self.drop(x)

        return x