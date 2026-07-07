import torch
import torch.nn as nn


class GNKAN(nn.Module):
    """
    Pure PyTorch GN-KAN / GR-KAN FFN block.

    Input:
        (B, C, H, W)

    Output:
        (B, C, H, W)

    Components:
        1. Pointwise expansion
        2. Group scaling
        3. GELU activation
        4. Depthwise spatial interaction
        5. Dynamic gating
        6. Pointwise projection
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = None,
        groups: int = 8,
        drop: float = 0.0,
        spatial_kernel: int = 3,
        use_gating: bool = True,
    ):
        super().__init__()

        hidden_channels = hidden_channels or (channels * 4)

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.groups = max(1, groups)
        self.use_gating = use_gating
        self.spatial_kernel = spatial_kernel

        self.drop = (
            nn.Dropout2d(drop)
            if drop > 0.0
            else nn.Identity()
        )

        # Expansion projection
        self.fc1 = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=True
        )

        # Reconstruction projection
        self.fc2 = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=True
        )

        # Depthwise spatial interaction
        if spatial_kernel > 1:
            pad = (spatial_kernel - 1) // 2

            self.depthwise_spatial = nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=spatial_kernel,
                padding=pad,
                groups=hidden_channels,
                bias=True
            )
        else:
            self.depthwise_spatial = nn.Identity()

        # Group-wise scaling
        assert hidden_channels % self.groups == 0

        self.group_scale = nn.Parameter(
            torch.ones(
                self.groups,
                hidden_channels // self.groups
            )
        )

        # Activation
        self.act = nn.GELU()

        # Dynamic gate
        if use_gating:
            self.gate = nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=1,
                    bias=True
                ),
                nn.Sigmoid()
            )
        else:
            self.gate = None

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

        if isinstance(
            self.depthwise_spatial,
            nn.Conv2d
        ):
            nn.init.kaiming_normal_(
                self.depthwise_spatial.weight,
                mode="fan_in",
                nonlinearity="relu"
            )

            nn.init.zeros_(
                self.depthwise_spatial.bias
            )

    def forward(self, x):

        B, C, H, W = x.shape

        # 1. Expansion
        h = self.fc1(x)

        # 2. Group Scaling
        if self.groups > 1:

            ch_per = (
                self.hidden_channels
                // self.groups
            )

            h = h.view(
                B,
                self.groups,
                ch_per,
                H,
                W
            )

            h = (
                h
                * self.group_scale.view(
                    1,
                    self.groups,
                    ch_per,
                    1,
                    1
                )
            )

            h = h.view(
                B,
                self.hidden_channels,
                H,
                W
            )

        # 3. Activation
        h = self.act(h)

        # 4. Spatial Interaction
        h = self.depthwise_spatial(h)

        # 5. Dynamic Gating
        if self.gate is not None:

            gate = self.gate(h)

            h = h * gate

        h = self.drop(h)

        # 6. Projection
        out = self.fc2(h)

        out = self.drop(out)

        return out