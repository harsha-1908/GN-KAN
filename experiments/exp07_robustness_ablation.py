import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

from common.ablation_factory import build_variant
from common.metrics import classification_metrics
from common.utils import get_device, create_exp_dir


# =====================================================
# CONFIG
# =====================================================

EXP_NAME = "EXP07_ROBUSTNESS_ABLATION"

SAVE_DIR = os.path.join(
    "outputs",
    EXP_NAME
)

create_exp_dir(SAVE_DIR)

DEVICE = get_device()

CHECKPOINT_ROOT = (
    "outputs/EXP04_BRAINTUMOR_ABLATION_100E"
)

VAL_DIR = (
    "/scratch/ghthilak_iitp/datasets/"
    "brisc2025/classification_task/val"
)

NOISE_LEVELS = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.05
]

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


# =====================================================
# NETWORK
# =====================================================

class BrainTumorNet(nn.Module):

    def __init__(self, variant):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.GELU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.GELU(),

            nn.MaxPool2d(2)
        )

        self.block = build_variant(
            variant,
            channels=64,
            hidden=256
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            64,
            4
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.block(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


# =====================================================
# NOISE TRANSFORM
# =====================================================

class AddGaussianNoise:

    def __init__(self, std):

        self.std = std

    def __call__(self, tensor):

        if self.std == 0:
            return tensor

        noise = torch.randn_like(
            tensor
        ) * self.std

        tensor = tensor + noise

        tensor = torch.clamp(
            tensor,
            0,
            1
        )

        return tensor


# =====================================================
# EVALUATION
# =====================================================

results = []

for noise_std in NOISE_LEVELS:

    print("\n")
    print("=" * 80)
    print(f"Noise Level = {noise_std}")
    print("=" * 80)

    transform = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        AddGaussianNoise(
            noise_std
        )
    ])

    val_set = torchvision.datasets.ImageFolder(
        VAL_DIR,
        transform=transform
    )

    val_loader = DataLoader(

        val_set,

        batch_size=128,

        shuffle=False,

        num_workers=8,

        pin_memory=True
    )

    for model_name in MODELS:

        print(
            f"Evaluating {model_name}"
        )

        model = BrainTumorNet(
            model_name
        ).to(DEVICE)

        ckpt = os.path.join(
            CHECKPOINT_ROOT,
            model_name,
            "best_model.pth"
        )

        model.load_state_dict(

            torch.load(
                ckpt,
                map_location=DEVICE
            )
        )

        model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    DEVICE
                )

                outputs = model(
                    images
                )

                pred = outputs.argmax(1)

                y_true.extend(
                    labels.numpy()
                )

                y_pred.extend(
                    pred.cpu().numpy()
                )

        metrics = classification_metrics(
            y_true,
            y_pred
        )

        row = {

            "Model":
            model_name,

            "NoiseSTD":
            noise_std,

            "Accuracy":
            metrics["Accuracy"],

            "Precision":
            metrics["Precision"],

            "Recall":
            metrics["Recall"],

            "F1":
            metrics["F1"]
        }

        results.append(row)

        pd.DataFrame(
            results
        ).to_csv(

            os.path.join(
                SAVE_DIR,
                "robustness_results.csv"
            ),

            index=False
        )


# =====================================================
# ROBUSTNESS DROP
# =====================================================

df = pd.DataFrame(results)

clean = df[
    df["NoiseSTD"] == 0.0
][["Model", "Accuracy"]]

clean.columns = [
    "Model",
    "CleanAcc"
]

df = df.merge(
    clean,
    on="Model"
)

df["RobustnessDrop"] = (
    df["CleanAcc"]
    - df["Accuracy"]
)

df.to_csv(

    os.path.join(
        SAVE_DIR,
        "robustness_results.csv"
    ),

    index=False
)

with open(

    os.path.join(
        SAVE_DIR,
        "robustness_summary.txt"
    ),

    "w"

) as f:

    f.write(
        df.to_string()
    )

print("\n")
print("=" * 80)
print("ROBUSTNESS EXPERIMENT COMPLETED")
print("=" * 80)