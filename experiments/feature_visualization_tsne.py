import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =====================================================
# MODIFY THESE
# =====================================================

DEVICE = "cuda"

DATASET_NAME = "BrainTumor"      # BrainTumor or CIFAR100

MLP_CKPT = "/home/ghthilak_iitp/gnkan_vs_kan_vs_mlp/outputs/EXP04_BRAINTUMOR_ABLATION_100E/MLP/best_model.pth"
KAN_CKPT = "/home/ghthilak_iitp/gnkan_vs_kan_vs_mlp/outputs/EXP04_BRAINTUMOR_ABLATION_100E/EfficientKAN/best_model.pth"
GNKAN_CKPT = "/home/ghthilak_iitp/gnkan_vs_kan_vs_mlp/outputs/EXP04_BRAINTUMOR_ABLATION_100E/GNKAN/best_model.pth"

SAVE_DIR = "/home/ghthilak_iitp/gnkan_vs_kan_vs_mlp/outputs/feature_visualization_tsne"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# =====================================================
# IMPORT YOUR DATASET + MODEL HERE
# =====================================================

from your_dataset_file import get_test_loader
from your_model_file import Net

# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(
    model,
    loader,
    max_samples=5000
):

    model.eval()

    features = []
    labels = []

    with torch.no_grad():

        for images, target in loader:

            images = images.to(DEVICE)

            feat = model.extract_features(images)

            features.append(
                feat.cpu().numpy()
            )

            labels.append(
                target.numpy()
            )

            if sum(len(x) for x in labels) >= max_samples:
                break

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    return features[:max_samples], labels[:max_samples]


# =====================================================
# PLOT FUNCTION
# =====================================================

def plot_tsne(
    features,
    labels,
    title,
    save_path
):

    print(f"Running t-SNE for {title}")

    # PCA before t-SNE
    pca = PCA(
        n_components=min(
            50,
            features.shape[1]
        )
    )

    features = pca.fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca"
    )

    emb = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        emb[:,0],
        emb[:,1],
        c=labels,
        s=8,
        alpha=0.7
    )

    plt.title(title)

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=400
    )

    plt.close()


# =====================================================
# LOAD DATA
# =====================================================

loader = get_test_loader(
    DATASET_NAME
)

# =====================================================
# LOAD MODELS
# =====================================================

models = {

    "MLP": MLP_CKPT,
    "EfficientKAN": KAN_CKPT,
    "GNKAN": GNKAN_CKPT
}

# =====================================================
# GENERATE INDIVIDUAL PLOTS
# =====================================================

all_embeddings = {}

for name, ckpt in models.items():

    print(
        f"\nLoading {name}"
    )

    model = Net(
        variant=name
    )

    model.load_state_dict(

        torch.load(
            ckpt,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    features, labels = extract_features(
        model,
        loader
    )

    all_embeddings[name] = (
        features,
        labels
    )

    plot_tsne(
        features,
        labels,
        f"{DATASET_NAME} - {name}",
        os.path.join(
            SAVE_DIR,
            f"{DATASET_NAME}_{name}_tsne.png"
        )
    )

# =====================================================
# COMBINED FIGURE
# =====================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,5)
)

for ax, name in zip(
    axes,
    ["MLP","EfficientKAN","GNKAN"]
):

    features, labels = all_embeddings[name]

    pca = PCA(
        n_components=min(
            50,
            features.shape[1]
        )
    )

    features = pca.fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca"
    )

    emb = tsne.fit_transform(features)

    ax.scatter(
        emb[:,0],
        emb[:,1],
        c=labels,
        s=5
    )

    ax.set_title(name)

    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

plt.savefig(

    os.path.join(
        SAVE_DIR,
        f"{DATASET_NAME}_comparison.png"
    ),

    dpi=500
)

print("\nDone.")