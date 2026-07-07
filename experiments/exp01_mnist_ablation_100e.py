import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

import time
import pandas as pd

import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

from common.utils import (
    create_exp_dir,
    count_params,
    get_device
)

from common.logger import (
    ExperimentLogger
)

from common.trainer import (
    Trainer
)

from common.ablation_factory import (
    build_variant
)


# =====================================================
# CONFIG
# =====================================================

EXP_NAME = "EXP01_MNIST_ABLATION_100E"

SAVE_DIR = os.path.join(
    "outputs",
    EXP_NAME
)

create_exp_dir(
    SAVE_DIR
)

logger = ExperimentLogger(
    SAVE_DIR
)

DEVICE = get_device()

logger.header(
    EXP_NAME
)


# =====================================================
# DATA
# =====================================================

transform = transforms.ToTensor()

MNIST_ROOT = "/home/ghthilak_iitp/gnkan_vs_mlp/data"

train_set = torchvision.datasets.MNIST(
    root=MNIST_ROOT,
    train=True,
    download=False,
    transform=transform
)

test_set = torchvision.datasets.MNIST(
    root=MNIST_ROOT,
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

test_loader = DataLoader(
    test_set,
    batch_size=256,
    shuffle=False,
    num_workers=4
)


# =====================================================
# NETWORK
# =====================================================

class MNISTNet(nn.Module):

    def __init__(self, variant):

        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=1
            ),
            nn.GELU()
        )

        self.block = build_variant(
            variant,
            channels=32,
            hidden=128
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                32 * 28 * 28,
                10
            )
        )

    def forward(self, x):

        x = self.embed(x)

        x = self.block(x)

        x = self.classifier(x)

        return x


# =====================================================
# MODELS
# =====================================================

models = [

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

results = []

summary_file = os.path.join(
    SAVE_DIR,
    "summary.txt"
)

with open(summary_file, "w") as f:

    f.write(
        f"{EXP_NAME}\n\n"
    )

for model_name in models:

    logger.write("\n")
    logger.write("=" * 80)
    logger.write(
        f"Training {model_name}"
    )
    logger.write("=" * 80)

    model = MNISTNet(
        model_name
    ).to(DEVICE)

    params = count_params(
        model
    )

    model_dir = os.path.join(
        SAVE_DIR,
        model_name
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    trainer = Trainer(
        model,
        DEVICE,
        logger,
        save_dir=model_dir
    )
    train_info = trainer.train(
        train_loader,
        test_loader,
        epochs=100,
        lr=1e-3
    )

    metrics = trainer.evaluate(
        test_loader
    )

    row = {

        "Model":
        model_name,

        "Accuracy":
        metrics["Accuracy"],

        "Precision":
        metrics["Precision"],

        "Recall":
        metrics["Recall"],

        "F1":
        metrics["F1"],

        "Params":
        params,

        "TrainTime":
        train_info["TrainTime"],

        "BestAccuracy":
        train_info["BestAccuracy"],

        "BestEpoch":
        train_info["BestEpoch"],

        "InferTime":
        metrics["InferTime"]
    }

    results.append(row)

    pd.DataFrame(
        results
    ).to_csv(

        os.path.join(
            SAVE_DIR,
            "results.csv"
        ),

        index=False
    )

    with open(
        summary_file,
        "a"
    ) as f:

        f.write(
            str(row)
            + "\n"
        )

# =====================================================
# FINAL
# =====================================================

df = pd.DataFrame(
    results
)

df = df.sort_values(
    "Accuracy",
    ascending=False
)

print("\n")
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(df)

df.to_csv(

    os.path.join(
        SAVE_DIR,
        "results.csv"
    ),

    index=False
)

logger.write("\n")
logger.write("=" * 80)
logger.write("EXPERIMENT COMPLETED")
logger.write("=" * 80)