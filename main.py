import enum
from backports.strenum import StrEnum

enum.StrEnum = StrEnum

import time
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import zadu

try:
    import pynndescent
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False
    from sklearn.neighbors import NearestNeighbors

def compute_distance_correlation(X, Y, max_samples=2000):
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sub, Y_sub = X[idx], Y[idx]
    else:
        X_sub, Y_sub = X, Y

    d_high = pdist(X_sub, metric='euclidean')
    d_low = pdist(Y_sub, metric='euclidean')

    return np.corrcoef(d_high, d_low)[0, 1]


def compute_kruskals_stress(X, Y, max_samples=2000):
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sub, Y_sub = X[idx], Y[idx]
    else:
        X_sub, Y_sub = X, Y

    d_high = pdist(X_sub, metric='euclidean')
    d_low = pdist(Y_sub, metric='euclidean')

    stress = np.sqrt(np.sum((d_high - d_low) ** 2) / np.sum(d_high ** 2))
    return stress


class MultilevelTSNE:
    def __init__(self, n_components=2, perplexity=None, n_levels=None, base_iterations=300, refine_iterations=250):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_levels = n_levels
        self.base_iterations = base_iterations
        self.refine_iterations = max(250, refine_iterations)

    def _build_knn_graph(self, X, k=30):
        n_samples = X.shape[0]
        if HAS_PYNNDESCENT and n_samples > 2000:
            index = pynndescent.NNDescent(X, n_neighbors=k + 1, n_jobs=-1)
            indices, _ = index.query(X, k=k + 1)
            indices = indices[:, 1:]
        else:
            nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
            nn.fit(X)
            _, indices = nn.kneighbors(X)

        row_indices = np.repeat(np.arange(n_samples), k)
        col_indices = indices.flatten()
        data = np.ones(n_samples * k, dtype=np.float32)

        W_sparse = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
        return (W_sparse + W_sparse.T) / 2.0

    def _coarsen_heavy_edge_matching(self, W, X_curr):
        n_nodes = W.shape[0]
        visited = np.zeros(n_nodes, dtype=bool)
        parent_mapping = np.zeros(n_nodes, dtype=int)

        coarse_id = 0
        W_coo = W.tocoo()

        sorted_indices = np.argsort(-W_coo.data)
        for idx in sorted_indices:
            u, v = W_coo.row[idx], W_coo.col[idx]
            if not visited[u] and not visited[v] and u != v:
                visited[u] = visited[v] = True
                parent_mapping[u] = parent_mapping[v] = coarse_id
                coarse_id += 1

        for node in range(n_nodes):
            if not visited[node]:
                parent_mapping[node] = coarse_id
                coarse_id += 1

        P = sp.csr_matrix((np.ones(n_nodes), (parent_mapping, np.arange(n_nodes))), shape=(coarse_id, n_nodes))
        W_coarse = P.dot(W).dot(P.T)

        degree = np.array(P.sum(axis=1)).flatten()
        degree[degree == 0] = 1.0
        P_norm = sp.diags(1.0 / degree).dot(P)
        X_coarse = P_norm.dot(X_curr)

        return W_coarse, parent_mapping, X_coarse

    def fit_transform(self, X):
        N = X.shape[0]
        if self.perplexity is None:
            self.perplexity = int(np.clip(np.sqrt(N), 15, 50))
        if self.n_levels is None:
            target_base_size = 400
            self.n_levels = max(1, int(np.round(np.log2(N / target_base_size))))

        G0 = self._build_knn_graph(X, k=self.perplexity)
        graph_hierarchy, mapping_hierarchy, feature_hierarchy = [G0], [], [X]

        curr_G, curr_X = G0, X
        for l in range(self.n_levels):
            curr_G, mapping, curr_X = self._coarsen_heavy_edge_matching(curr_G, curr_X)
            graph_hierarchy.append(curr_G)
            mapping_hierarchy.append(mapping)
            feature_hierarchy.append(curr_X)

        coarsest_G, coarsest_X = graph_hierarchy[-1], feature_hierarchy[-1]
        base_perp = min(15, max(5, coarsest_G.shape[0] - 2))
        base_tsne = TSNE(
            n_components=self.n_components, perplexity=base_perp,
            max_iter=self.base_iterations, init='random',
            method='barnes_hut', random_state=42
        )
        Y_current = base_tsne.fit_transform(coarsest_X)

        for l in reversed(range(self.n_levels)):
            mapping = mapping_hierarchy[l]
            finer_X = feature_hierarchy[l]
            n_finer = finer_X.shape[0]

            Y_interpolated = np.zeros((n_finer, self.n_components), dtype=np.float32)
            for child_idx in range(n_finer):
                Y_interpolated[child_idx] = Y_current[mapping[child_idx]] + np.random.normal(0, 0.5, size=self.n_components)

            Y_interpolated = (Y_interpolated - np.mean(Y_interpolated, axis=0)) / (np.std(Y_interpolated, axis=0) + 1e-5) * 1e-4
            iters = 500 if l == 0 else self.refine_iterations

            refine_tsne = TSNE(
                n_components=self.n_components, perplexity=self.perplexity,
                max_iter=iters, init=Y_interpolated,
                learning_rate='auto', method='barnes_hut', random_state=42
            )
            Y_current = refine_tsne.fit_transform(finer_X)

        return Y_current


def evaluate_and_plot(name, X, y):
    print(f"\nEvaluating Multilevel vs Baseline on {name}...")

    # 1. Run Multilevel t-SNE
    t0 = time.time()
    ml_tsne = MultilevelTSNE()
    Y_multi = ml_tsne.fit_transform(X)
    t_multi = time.time() - t0

    # 2. Run Standard t-SNE Baseline
    t0 = time.time()
    baseline = TSNE(n_components=2, perplexity=30, max_iter=1000, init='random', method='barnes_hut', random_state=42)
    Y_base = baseline.fit_transform(X)
    t_base = time.time() - t0

    # 3. Compute Local Metrics via ZADU
    spec_local = [{"id": "tnc", "params": {"k": 10}}]
    eval_multi_local = zadu.ZADU(spec_local, X).measure(Y_multi)[0]
    eval_base_local = zadu.ZADU(spec_local, X).measure(Y_base)[0]

    # 4. Compute Global Metrics
    dcor_multi = compute_distance_correlation(X, Y_multi)
    dcor_base = compute_distance_correlation(X, Y_base)

    stress_multi = compute_kruskals_stress(X, Y_multi)
    stress_base = compute_kruskals_stress(X, Y_base)

    # Print Quantitative Results Table
    print("\n" + "=" * 80)
    print(f"QUANTITATIVE RESULTS: {name}")
    print("=" * 80)
    print(f"{'Method':<20} | {'Time (s)':<8} | {'Trust. (↑)':<10} | {'Cont. (↑)':<10} | {'DistCorr (↑)':<12} | {'Stress (↓)':<10}")
    print("-" * 80)
    print(f"{'Multilevel t-SNE':<20} | {t_multi:<8.2f} | {eval_multi_local['trustworthiness']:<10.4f} | {eval_multi_local['continuity']:<10.4f} | {dcor_multi:<12.4f} | {stress_multi:<10.4f}")
    print(f"{'Baseline t-SNE':<20} | {t_base:<8.2f} | {eval_base_local['trustworthiness']:<10.4f} | {eval_base_local['continuity']:<10.4f} | {dcor_base:<12.4f} | {stress_base:<10.4f}")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    axes[0].scatter(Y_multi[:, 0], Y_multi[:, 1], c=y, cmap='Spectral', s=6, alpha=0.7)
    axes[0].set_title(f"{name} - Multilevel t-SNE ({t_multi:.2f}s)")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].scatter(Y_base[:, 0], Y_base[:, 1], c=y, cmap='Spectral', s=6, alpha=0.7)
    axes[1].set_title(f"{name} - Standard t-SNE Baseline ({t_base:.2f}s)")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{name}_benchmark.png", dpi=300)
    plt.close()
    print(f"Saved visualization to '{name}_benchmark.png'")


def load_dataset(dataset_name, n_samples=None):
    if dataset_name == "COIL-20":
        data = fetch_openml(data_id=46783, as_frame=False, parser='liac-arff')
    elif dataset_name == "USPS":
        data = fetch_openml(data_id=41082, as_frame=False, parser='liac-arff')
    elif dataset_name == "Pendigits":
        data = fetch_openml('pendigits', version=1, as_frame=False, parser='liac-arff')
    elif dataset_name == "MNIST":
        data = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X = data.data.astype(np.float32)
    y = data.target.astype(int)

    if n_samples is not None and n_samples < X.shape[0]:
        X, y = X[:n_samples], y[:n_samples]

    return X, y


if __name__ == "__main__":
    datasets_to_test = [
        ("COIL-20", None),
        ("Pendigits", None),
        ("MNIST", 10000),
    ]

    for name, n_samples in datasets_to_test:
        X, y = load_dataset(name, n_samples)
        evaluate_and_plot(name, X, y)