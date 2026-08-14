import enum
from backports.strenum import StrEnum

enum.StrEnum = StrEnum

import os
import time
import numpy as np
import scipy.sparse as sp
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml, load_digits
import matplotlib.pyplot as plt
import zadu


class DRBenchmarkLoader:
    def __init__(self, download_dir="./benchmark_data"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def load_dataset_from_npz(self, file_path):
        data = np.load(file_path)
        X = data['X'].astype(np.float32)
        y = data['y'] if 'y' in data else None
        return X, y


class MultilevelTSNE:
    def __init__(self, n_components=2, perplexity=None, n_levels=None, base_iterations=300, refine_iterations=250):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_levels = n_levels
        self.base_iterations = base_iterations
        self.refine_iterations = max(250, refine_iterations)

    def _build_knn_graph(self, X, k=30):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        n_samples = X.shape[0]
        row_indices = np.repeat(np.arange(n_samples), k)
        col_indices = indices.flatten()
        data = np.ones(n_samples * k, dtype=np.float32)

        W_sparse = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
        W_symmetric = (W_sparse + W_sparse.T) / 2.0
        return W_symmetric

    def _coarsen_heavy_edge_matching(self, W):
        n_nodes = W.shape[0]
        visited = np.zeros(n_nodes, dtype=bool)
        parent_mapping = np.zeros(n_nodes, dtype=int)

        coarse_id = 0
        W_coo = W.tocoo()

        sorted_indices = np.argsort(-W_coo.data)
        for idx in sorted_indices:
            u, v = W_coo.row[idx], W_coo.col[idx]
            if not visited[u] and not visited[v] and u != v:
                visited[u] = True
                visited[v] = True
                parent_mapping[u] = coarse_id
                parent_mapping[v] = coarse_id
                coarse_id += 1

        for node in range(n_nodes):
            if not visited[node]:
                parent_mapping[node] = coarse_id
                coarse_id += 1

        P = sp.csr_matrix((np.ones(n_nodes), (parent_mapping, np.arange(n_nodes))), shape=(coarse_id, n_nodes))
        W_coarse = P.dot(W).dot(P.T)

        return W_coarse, parent_mapping

    def fit_transform(self, X):
        N = X.shape[0]

        if self.perplexity is None:
            self.perplexity = int(np.clip(np.sqrt(N), 15, 50))

        if self.n_levels is None:
            target_base_size = 400
            self.n_levels = max(1, int(np.round(np.log2(N / target_base_size))))

        print(f"[Phase 1] Constructing k-NN Similarity Graph for N={X.shape[0]}...")
        G0 = self._build_knn_graph(X, k=self.perplexity)

        graph_hierarchy = [G0]
        mapping_hierarchy = []

        curr_G = G0
        for l in range(self.n_levels):
            print(f"[Phase 2] Coarsening Level {l} -> {l + 1} (Nodes: {curr_G.shape[0]})...")
            curr_G, mapping = self._coarsen_heavy_edge_matching(curr_G)
            graph_hierarchy.append(curr_G)
            mapping_hierarchy.append(mapping)

        coarsest_G = graph_hierarchy[-1]
        print(f"[Phase 3] Optimizing Coarsest Base Graph GL (Nodes: {coarsest_G.shape[0]})...")

        base_tsne = TSNE(
            n_components=self.n_components,
            perplexity=min(15, coarsest_G.shape[0] - 1),
            max_iter=self.base_iterations,
            init='random',
            random_state=42
        )
        Y_coarse = base_tsne.fit_transform(coarsest_G.toarray())

        Y_current = Y_coarse
        for l in reversed(range(self.n_levels)):
            mapping = mapping_hierarchy[l]
            finer_G = graph_hierarchy[l]
            print(f"[Phase 3] Uncoarsening Level {l + 1} -> {l} (Refining {finer_G.shape[0]} nodes)...")

            n_finer = finer_G.shape[0]
            Y_interpolated = np.zeros((n_finer, self.n_components), dtype=np.float32)
            for child_idx in range(n_finer):
                parent_idx = mapping[child_idx]
                jitter = np.random.normal(0, 0.5, size=self.n_components)
                Y_interpolated[child_idx] = Y_current[parent_idx] + jitter

            # Standardize and scale coordinates so t-SNE forces spread out properly
            Y_interpolated = (Y_interpolated - np.mean(Y_interpolated, axis=0)) / (np.std(Y_interpolated, axis=0) + 1e-5) * 1e-4

            # Set iterations: full level (l=0) gets more steps to relax clusters
            iters = 500 if l == 0 else self.refine_iterations

            refine_tsne = TSNE(
                n_components=self.n_components,
                perplexity=self.perplexity,
                max_iter=iters,
                init=Y_interpolated,
                learning_rate='auto',
                random_state=42
            )
            Y_current = refine_tsne.fit_transform(X[:n_finer])

        return Y_current

def run_evaluation_experiment(X, Y_multilevel, Y_baseline):
    print("\nRunning Quantitative Evaluation...")

    spec_local = [{"id": "tnc", "params": {"k": 10}}]

    scores_multi = zadu.ZADU(spec_local, X).measure(Y_multilevel)
    scores_base = zadu.ZADU(spec_local, X).measure(Y_baseline)

    tnc_multi = scores_multi[0]
    tnc_base = scores_base[0]

    print(
        f"Multilevel t-SNE -> Trustworthiness: {tnc_multi['trustworthiness']:.4f} | Continuity: {tnc_multi['continuity']:.4f}")
    print(
        f"Baseline t-SNE   -> Trustworthiness: {tnc_base['trustworthiness']:.4f} | Continuity: {tnc_base['continuity']:.4f}")


def visualize_embeddings(Y_proposed, Y_baseline, labels):
    print("\nGenerating visualization plots...")

    if labels is None:
        c = 'steelblue'
        cmap = None
    else:
        c = labels
        cmap = 'Spectral'

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)

    scatter1 = axes[0].scatter(Y_proposed[:, 0], Y_proposed[:, 1], c=c, cmap=cmap, s=5, alpha=0.7)
    axes[0].set_title("Proposed Multilevel t-SNE Embedding", fontsize=14)
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    scatter2 = axes[1].scatter(Y_baseline[:, 0], Y_baseline[:, 1], c=c, cmap=cmap, s=5, alpha=0.7)
    axes[1].set_title("Standard t-SNE Baseline", fontsize=14)
    axes[1].set_xlabel("Dimension 1")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Call tight_layout BEFORE adding the colorbar to avoid warning
    plt.tight_layout()

    if labels is not None:
        cbar = fig.colorbar(scatter2, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
        cbar.set_label('True Class Label', rotation=270, labelpad=15)

    plt.show()

def download_osf_dataset(dataset_name="Fashion-MNIST"):
    print(f"Fetching benchmark dataset ({dataset_name})...")

    try:
        print("Downloading 'Fashion-MNIST' from OpenML...")
        mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='liac-arff')
        X = mnist.data[:2500].astype(np.float32)
        y = mnist.target[:2500].astype(int)
        print(f"Successfully loaded 'Fashion-MNIST': Shape = {X.shape}")
        return X, y

    except Exception as e:
        print(f"Could not reach OpenML ({e}). Falling back to Scikit-Learn 'digits' dataset...")
        digits = load_digits()
        X = digits.data.astype(np.float32)
        y = digits.target
        print(f"Successfully loaded 'digits': Shape = {X.shape}")
        return X, y

# def download_osf_dataset(dataset_name="coil-20"):
#     print(f"Fetching benchmark dataset ({dataset_name})...")
#     try:
#         coil = fetch_openml(data_id=46783, as_frame=False, parser='liac-arff')
#         X = coil.data.astype(np.float32)
#         y = coil.target.astype(int)
#         print(f"Successfully loaded 'COIL-20': Shape = {X.shape}")
#         return X, y
#     except Exception as e:
#         print(f"Failed to fetch COIL-20: {e}")
#         raise e

# def download_osf_dataset(dataset_name="USPS"):
#     print(f"Fetching benchmark dataset ({dataset_name})...")
#     try:
#         usps = fetch_openml(data_id=41082, as_frame=False, parser='liac-arff')
#         X = usps.data[:2500].astype(np.float32)
#         y = usps.target[:2500].astype(int)
#         print(f"Successfully loaded 'USPS': Shape = {X.shape}")
#         return X, y
#     except Exception as e:
#         print(f"Failed to fetch USPS: {e}")
#         raise e

# def download_osf_dataset(dataset_name="MNIST"):
#     print(f"Fetching benchmark dataset ({dataset_name})...")
#     try:
#         mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
#         X = mnist.data[:2500].astype(np.float32)
#         y = mnist.target[:2500].astype(int)
#         print(f"Successfully loaded 'MNIST': Shape = {X.shape}")
#         return X, y
#     except Exception as e:
#         print(f"Failed to fetch MNIST: {e}")
#         raise e

if __name__ == "__main__":
    X, y = download_osf_dataset(dataset_name="Fashion-MNIST")

    print("\nStarting Multilevel t-SNE...")
    t0 = time.time()
    ml_tsne = MultilevelTSNE(
        n_levels=2,
        perplexity=30,
        base_iterations=300,
        refine_iterations=250
    )
    Y_proposed = ml_tsne.fit_transform(X)
    multilevel_time = time.time() - t0
    print(f"Multilevel t-SNE completed in {multilevel_time:.2f} seconds.")

    print("\nStarting Standard t-SNE Baseline...")
    t0 = time.time()
    standard_tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        init='random',
        random_state=42
    )
    Y_baseline = standard_tsne.fit_transform(X)
    baseline_time = time.time() - t0
    print(f"Standard t-SNE completed in {baseline_time:.2f} seconds.")

    run_evaluation_experiment(X, Y_proposed, Y_baseline)

    visualize_embeddings(Y_proposed, Y_baseline, y)