import os
import sys
import numpy as np
import random

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

from models.gnkan import GNKAN
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

# =====================================================
# CONFIG
# =====================================================

EXP_NAME = "GROUP_ABLATION_CIFAR100"

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

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    ),
])

DATA_ROOT = "/home/ghthilak_iitp/gnkan_vs_kan_vs_mlp/data"

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

class CIFARNet(nn.Module):

    def __init__(
        self,
        groups,
        num_classes=100
    ):

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.block1 = GNKAN(
            channels=64,
            hidden_channels=512,
            groups=groups
        )

        self.block2 = GNKAN(
            channels=64,
            hidden_channels=512,
            groups=groups
        )

        self.block3 = GNKAN(
            channels=64,
            hidden_channels=512,
            groups=groups
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.stem(x)

        x = x + self.block1(x)

        x = x + self.block2(x)

        x = x + self.block3(x)

        x = self.head(x)

        return x

# =====================================================
# MODELS
# =====================================================

SEEDS = [
    42,
    43,
    44
]

GROUPS = [
    8,
    16,
    32
]


def set_seed(seed):

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


summary_file = os.path.join(
    SAVE_DIR,
    "summary.txt"
)

results = []

with open(summary_file, "w") as f:

    f.write(
        f"{EXP_NAME}\n\n"
    )


for g in GROUPS:

    logger.write("\n")
    logger.write("=" * 80)
    logger.write(
        f"GROUP G = {g}"
    )
    logger.write("=" * 80)

    group_runs = []

    for seed in SEEDS:

        set_seed(seed)

        logger.write(
            f"\nTraining GNKAN_G{g}_Seed{seed}"
        )

        model = CIFARNet(
            groups=g,
            num_classes=100
        ).to(DEVICE)

        params = count_params(
            model
        )

        model_dir = os.path.join(
            SAVE_DIR,
            f"G{g}_Seed{seed}"
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

        run_row = {

            "Seed":
            seed,

            "Accuracy":
            metrics["Accuracy"],

            "Precision":
            metrics["Precision"],

            "Recall":
            metrics["Recall"],

            "F1":
            metrics["F1"],

            "TrainTime":
            train_info["TrainTime"],

            "BestAccuracy":
            train_info["BestAccuracy"],

            "BestEpoch":
            train_info["BestEpoch"],

            "InferTime":
            metrics["InferTime"]
        }

        group_runs.append(
            run_row
        )

    accs = [
        r["Accuracy"]
        for r in group_runs
    ]

    precs = [
        r["Precision"]
        for r in group_runs
    ]

    recs = [
        r["Recall"]
        for r in group_runs
    ]

    f1s = [
        r["F1"]
        for r in group_runs
    ]

    best_accs = [
        r["BestAccuracy"]
        for r in group_runs
    ]

    best_epochs = [
        r["BestEpoch"]
        for r in group_runs
    ]

    train_times = [
        r["TrainTime"]
        for r in group_runs
    ]

    infer_times = [
        r["InferTime"]
        for r in group_runs
    ]

    summary_row = {

        "Groups":
        g,

        "AccuracyMean":
        np.mean(accs),

        "AccuracyStd":
        np.std(accs),

        "PrecisionMean":
        np.mean(precs),

        "PrecisionStd":
        np.std(precs),

        "RecallMean":
        np.mean(recs),

        "RecallStd":
        np.std(recs),

        "F1Mean":
        np.mean(f1s),

        "F1Std":
        np.std(f1s),

        "BestAccuracyMean":
        np.mean(best_accs),

        "BestAccuracyStd":
        np.std(best_accs),

        "BestEpochMean":
        np.mean(best_epochs),

        "BestEpochStd":
        np.std(best_epochs),

        "TrainTimeMean":
        np.mean(train_times),

        "TrainTimeStd":
        np.std(train_times),

        "InferTimeMean":
        np.mean(infer_times),

        "InferTimeStd":
        np.std(infer_times),

        "Params":
        params
    }

    results.append(
        summary_row
    )
    with open(
        summary_file,
        "a"
    ) as f:

        f.write(
            str(summary_row)
            + "\n"
        )

    pd.DataFrame(
        results
    ).to_csv(

        os.path.join(
            SAVE_DIR,
            "group_scaling_results.csv"
        ),

        index=False
    )

    logger.write(
        f"\nG={g} | "
        f"Acc={summary_row['AccuracyMean']:.4f}+/-{summary_row['AccuracyStd']:.4f} | "
        f"BestAcc={summary_row['BestAccuracyMean']:.4f}+/-{summary_row['BestAccuracyStd']:.4f}"
    )

# =====================================================
# FINAL
# =====================================================

df = pd.DataFrame(
    results
)

df = df.sort_values(
    "AccuracyMean",
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
        "group_scaling_results.csv"
    ),

    index=False
)

logger.write("\n")
logger.write("=" * 80)
logger.write("EXPERIMENT COMPLETED")
logger.write("=" * 80)