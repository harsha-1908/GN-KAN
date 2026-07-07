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

EXP_NAME = "EXP15_SCALING_STUDY_BRAINTUMOR_BS512_RESUME"

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
# DATASET PATHS
# =====================================================

TRAIN_DIR = (
    "/scratch/ghthilak_iitp/datasets/"
    "brisc2025/classification_task/train"
)

VAL_DIR = (
    "/scratch/ghthilak_iitp/datasets/"
    "brisc2025/classification_task/val"
)


# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

#    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(
        5
    ),

    transforms.ToTensor()
])

val_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor()
])


# =====================================================
# DATASETS
# =====================================================

train_set = torchvision.datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_set = torchvision.datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

logger.write(
    f"Train Samples: {len(train_set)}"
)

logger.write(
    f"Validation Samples: {len(val_set)}"
)

logger.write(
    f"Classes: {len(train_set.classes)}"
)

logger.write(
    f"Class Names: {train_set.classes}"
)


# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(

    train_set,

    batch_size=128,

    shuffle=True,

    num_workers=8,

    pin_memory=True
)

val_loader = DataLoader(

    val_set,

    batch_size=128,

    shuffle=False,

    num_workers=8,

    pin_memory=True
)

# =====================================================
# NETWORK
# =====================================================

class BrainTumorNet(nn.Module):

    def __init__(
        self,
        variant,
        hidden_dim
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
            hidden=hidden_dim
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.fc = nn.Linear(
            64,
            4
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

#    "MLP",

    "EfficientKAN",

    "G",
    "S",
    "D",

    "GS",
    "GD",

    "SD",

    "GNKAN"
]

HIDDEN_DIMS = [

#    64,

#    128,

#    256,

    512
]

# =====================================================
# TRAINING LOOP
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

for hidden_dim in HIDDEN_DIMS:

    logger.write("\n")

    logger.write(
        "=" * 80
    )

    logger.write(
        f"HIDDEN DIM = {hidden_dim}"
    )

    logger.write(
        "=" * 80
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

        model = BrainTumorNet(
            model_name,
            hidden_dim
        ).to(DEVICE)

        params = count_params(
            model
        )

        model_dir = os.path.join(

            SAVE_DIR,

            f"hidden_{hidden_dim}",

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

            val_loader,

            epochs=50,

            lr=1e-3
        )

        metrics = trainer.evaluate(
            val_loader
        )

        row = {

            "HiddenDim":
            hidden_dim,

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
            train_info[
                "TrainTime"
            ],

            "BestAccuracy":
            train_info[
                "BestAccuracy"
            ],

            "BestEpoch":
            train_info[
                "BestEpoch"
            ],

            "InferTime":
            metrics[
                "InferTime"
            ]
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

    [
        "HiddenDim",
        "Accuracy"
    ],

    ascending=[
        True,
        False
    ]
)

print("\n")

print(
    "=" * 80
)

print(
    "FINAL RESULTS"
)

print(
    "=" * 80
)

print(df)

df.to_csv(

    os.path.join(
        SAVE_DIR,
        "results.csv"
    ),

    index=False
)

logger.write("\n")

logger.write(
    "=" * 80
)

logger.write(
    "EXPERIMENT COMPLETED"
)

logger.write(
    "=" * 80
)