import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Global settings
RNG_SEED = 42
OUT_DIR = "outputs"
DATA_PATH = "data/Mall_Customers.csv"   # <-- put your dataset here
os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset():
    """Load dataset from CSV. User must provide a dataset in data/ folder."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please add your CSV (e.g., mall_customers.csv)."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset from {DATA_PATH}, shape={df.shape}")

    # Drop non-numeric columns if any (like IDs or names)
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] < df.shape[1]:
        print("ℹ️ Dropped non-numeric columns for clustering.")

    return df_numeric


def elbow_method(X, max_k=10):
    """Run elbow method to determine optimal K."""
    inertia = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.figure()
    plt.plot(K_range, inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(K_range)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "elbow_method.png"), dpi=150)
    plt.close()
    print("📊 Elbow method plot saved -> outputs/elbow_method.png")


def fit_kmeans(X, k=3):
    """Fit KMeans and return labels & model."""
    kmeans = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def visualize_clusters(X, labels, title="KMeans Clusters"):
    """Visualize clusters in 2D using PCA if needed."""
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=RNG_SEED)
        X_vis = pca.fit_transform(X)
        print("ℹ️ Used PCA for 2D visualization.")
    else:
        X_vis = X

    plt.figure()
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap="viridis", s=40, edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1 (or PCA1)")
    plt.ylabel("Feature 2 (or PCA2)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "clusters.png"), dpi=150)
    plt.close()
    print("📊 Cluster visualization saved -> outputs/clusters.png")


def main():
    # ===== Step 1: Load dataset =====
    df = load_dataset()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # ===== Step 2: Elbow Method =====
    elbow_method(X_scaled, max_k=10)

    # ===== Step 3: Fit KMeans =====
    optimal_k = 3  # set manually after checking elbow plot
    labels, model = fit_kmeans(X_scaled, k=optimal_k)

    # ===== Step 4: Evaluate clustering =====
    sil_score = silhouette_score(X_scaled, labels)
    print(f"✅ Silhouette Score (k={optimal_k}): {sil_score:.4f}")

    # ===== Step 5: Visualize clusters =====
    visualize_clusters(X_scaled, labels, title=f"KMeans Clusters (k={optimal_k})")


if __name__ == "__main__":
    main()
