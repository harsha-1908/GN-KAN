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

EXP_NAME = "EXP06_CIFAR100_ABLATION_100E"

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

DATA_ROOT = (
    "/home/ghthilak_iitp/gnkan_vs_kan_vs_mlp/data"
)

transform_train = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor()
])

transform_test = transforms.Compose([

    transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR100(

    root=DATA_ROOT,

    train=True,

    download=False,

    transform=transform_train
)

test_set = torchvision.datasets.CIFAR100(

    root=DATA_ROOT,

    train=False,

    download=False,

    transform=transform_test
)

logger.write(
    f"Train Samples: {len(train_set)}"
)

logger.write(
    f"Test Samples: {len(test_set)}"
)

logger.write(
    "Classes: 100"
)

train_loader = DataLoader(

    train_set,

    batch_size=128,

    shuffle=True,

    num_workers=8,

    pin_memory=True
)

test_loader = DataLoader(

    test_set,

    batch_size=128,

    shuffle=False,

    num_workers=8,

    pin_memory=True
)

# =====================================================
# NETWORK
# =====================================================

class CIFAR100Net(nn.Module):

    def __init__(
        self,
        variant
    ):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.GELU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.GELU(),

            nn.MaxPool2d(
                2
            )
        )

        self.block = build_variant(

            variant,

            channels=64,

            hidden=256
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.fc = nn.Linear(
            64,
            100
        )

    def forward(
        self,
        x
    ):

        x = self.stem(x)

        x = self.block(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc(x)

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

# =====================================================
# TRAINING
# =====================================================

results = []

summary_file = os.path.join(
    SAVE_DIR,
    "summary.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        f"{EXP_NAME}\n\n"
    )

for model_name in models:

    logger.write("\n")

    logger.write(
        "=" * 80
    )

    logger.write(
        f"Training {model_name}"
    )

    logger.write(
        "=" * 80
    )

    model = CIFAR100Net(
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

    results.append(
        row
    )

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
# FINAL RESULTS
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