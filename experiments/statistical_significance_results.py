import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# ==========================================================
# ENTER SEED-WISE ACCURACIES HERE
# ==========================================================

results = {

    "MNIST": {
        "MLP":          [0.9319, 0.9362, 0.9338],
        "EfficientKAN": [0.9346, 0.9341, 0.9343],
        "GNKAN":        [0.982, 0.9824, 0.984],
    },

    "FashionMNIST": {
        "MLP":          [0.8644, 0.8626, 0.8615],
        "EfficientKAN": [0.8643, 0.8635, 0.8644],
        "GNKAN":        [0.9001, 0.8954, 0.9016],
    },

    "FashionMNIST-V2": {
        "MLP":          [0.9081, 0.9071, 0.9069],
        "EfficientKAN": [0.9076, 0.9086, 0.9082],
        "GNKAN":        [0.9178, 0.9227, 0.9187],
    },

    "CIFAR10": {
        "MLP":          [0.6495, 0.6468, 0.6539],
        "EfficientKAN": [0.6913, 0.6838, 0.6899],
        "GNKAN":        [0.7626, 0.7641, 0.7635],
    },

    "CIFAR100": {
        "MLP":          [0.52, 0.5125, 0.5121],
        "EfficientKAN": [0.5035, 0.4985, 0.4961],
        "GNKAN":        [0.5625, 0.5561, 0.5564],
    },

    "BrainTumor": {
        "MLP":          [0.855, 0.858, 0.844],
        "EfficientKAN": [0.845, 0.84, 0.831],
        "GNKAN":        [0.92, 0.916, 0.923],
    },
}

# ==========================================================
# STATISTICAL ANALYSIS
# ==========================================================

rows = []

for dataset, vals in results.items():

    mlp = np.array(vals["MLP"])
    kan = np.array(vals["EfficientKAN"])
    gn  = np.array(vals["GNKAN"])

    # paired t-tests
    _, p_mlp = ttest_rel(gn, mlp)
    _, p_kan = ttest_rel(gn, kan)

    # effect size (Cohen's d)
    diff_mlp = gn - mlp
    diff_kan = gn - kan

    d_mlp = np.mean(diff_mlp) / np.std(diff_mlp, ddof=1)
    d_kan = np.mean(diff_kan) / np.std(diff_kan, ddof=1)

    rows.append({
        "Dataset": dataset,

        "GNKAN Mean":
        f"{np.mean(gn)*100:.2f} +/- {np.std(gn,ddof=1)*100:.2f}",

        "MLP Mean":
        f"{np.mean(mlp)*100:.2f} +/- {np.std(mlp,ddof=1)*100:.2f}",

        "KAN Mean":
        f"{np.mean(kan)*100:.2f} +/- {np.std(kan,ddof=1)*100:.2f}",

        "p(GNKAN vs MLP)": p_mlp,

        "p(GNKAN vs KAN)": p_kan,

        "Cohen d vs MLP": d_mlp,

        "Cohen d vs KAN": d_kan
    })

df = pd.DataFrame(rows)

print("\n")
print("="*120)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("="*120)

print(df)

df.to_csv(
    "statistical_significance_results.csv",
    index=False
)

print("\nSaved: statistical_significance_results.csv")