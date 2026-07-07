import os
import time
import pandas as pd

import torch
import torch.nn as nn

from common.ablation_factory import build_variant
from common.utils import count_params


# =====================================================
# CONFIG
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [

    "MLP",

    "EfficientKAN",

    "G",
    "S",
    "D",

    "GS",
    "GD",

    "SD",

    "GNKAN"
]

SAVE_DIR = "outputs/EXP16_EFFICIENCY_ABLATION"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# =====================================================
# NETWORK
# =====================================================

class BenchmarkNet(nn.Module):

    def __init__(
        self,
        variant
    ):

        super().__init__()

        self.embed = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()
        )

        self.block = build_variant(

            variant,

            channels=64,

            hidden=256
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            64,
            10
        )

    def forward(
        self,
        x
    ):

        x = self.embed(x)

        x = self.block(x)

        x = self.pool(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x


# =====================================================
# BENCHMARK
# =====================================================

results = []

dummy = torch.randn(

    16,
    3,
    224,
    224

).to(DEVICE)

for model_name in MODELS:

    torch.cuda.empty_cache()

    print(
        f"\nBenchmarking {model_name}"
    )

    model = BenchmarkNet(
        model_name
    ).to(DEVICE)

    model.eval()

    params = count_params(
        model
    )

    model_size_mb = (

        sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )

        / 1024**2
    )

    # ------------------------------
    # warmup
    # ------------------------------

    with torch.no_grad():

        for _ in range(20):

            _ = model(
                dummy
            )

    torch.cuda.synchronize()

    # ------------------------------
    # latency
    # ------------------------------

    start = time.time()

    with torch.no_grad():

        for _ in range(100):

            _ = model(
                dummy
            )

    torch.cuda.synchronize()

    total_time = (
        time.time() - start
    )

    latency = (

        total_time

        / 100
    )

    throughput = (

        16 * 100

        / total_time
    )

    row = {

        "Model":
        model_name,

        "Params":
        params,

        "ModelSizeMB":
        round(
            model_size_mb,
            3
        ),

        "LatencySec":
        round(
            latency,
            6
        ),

        "ThroughputImgSec":
        round(
            throughput,
            2
        )
    }

    results.append(
        row
    )

    print(
        row
    )

# =====================================================
# SAVE
# =====================================================

df = pd.DataFrame(
    results
)

df.to_csv(

    os.path.join(
        SAVE_DIR,
        "efficiency_results.csv"
    ),

    index=False
)

print(df)