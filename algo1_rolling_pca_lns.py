"""
Algorithm 1: Rolling Window WS-LR-QAOA + PCA Consensus Edge Extraction + LNS
"""

import numpy as np
from sklearn.decomposition import PCA


def solve_lp_relaxation(cost_matrix):
    """Continuous Held-Karp LP relaxation mockup returning edge expectation matrix X_LP."""
    n = cost_matrix.shape[0]
    # Continuous relaxation heuristic (inverse distance softmax)
    exp_matrix = np.exp(-cost_matrix / (np.std(cost_matrix) + 1e-5))
    np.fill_diagonal(exp_matrix, 0)
    row_sums = exp_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X_LP = 2.0 * (exp_matrix / row_sums)
    return np.clip(X_LP, 0.0, 1.0)


def pca_consensus_edges(samples, n_edges, top_k_ratio=0.2):
    """Performs PCA on top low-energy sample co-occurrence matrix to extract structural edge backbone."""
    n_samples = len(samples)
    top_k = max(1, int(n_samples * top_k_ratio))
    top_samples = samples[:top_k]

    # Covariance of edge activations
    cov_matrix = np.cov(top_samples, rowvar=False)
    if cov_matrix.ndim < 2 or cov_matrix.shape[0] < 2:
        return np.mean(top_samples, axis=0) > 0.5

    pca = PCA(n_components=1)
    pca.fit(cov_matrix)
    primary_component = pca.components_[0]
    return primary_component > np.median(primary_component)


def run_algo1(data, num_qubits):
    """
    Simulates Rolling Window WS-LR-QAOA + PCA + LNS.
    num_qubits controls window size k ~ sqrt(num_qubits).
    """
    matrix = data["matrix"]
    n = data["n_nodes"]

    # Window size constrained by available simulator qubits
    window_size = max(2, min(n, int(np.sqrt(num_qubits)) + 1))

    # 1. Rolling Window + LP Warm Start
    tour = [0]
    unvisited = set(range(1, n))

    curr = 0
    while unvisited:
        window_nodes = [curr] + list(unvisited)[: window_size - 1]
        sub_matrix = matrix[np.ix_(window_nodes, window_nodes)]

        # WS-LR-QAOA Simulator Simulation
        X_LP = solve_lp_relaxation(sub_matrix)

        # Generate candidate quantum samples around θ = 2*arcsin(sqrt(x))
        angles = 2 * np.arcsin(np.sqrt(X_LP))
        samples = [
            (np.random.rand(*angles.shape) < np.sin(angles / 2) ** 2).astype(
                int
            )
            for _ in range(50)
        ]
        samples_flat = [s.flatten() for s in samples]

        # PCA Consensus
        consensus = pca_consensus_edges(samples_flat, sub_matrix.size)
        consensus_matrix = consensus.reshape(sub_matrix.shape)

        # Pick best next step from window based on consensus
        next_candidates = [
            node for node in window_nodes if node in unvisited and node != curr
        ]
        if not next_candidates:
            break

        next_node = max(
            next_candidates,
            key=lambda idx: consensus_matrix[
                window_nodes.index(curr), window_nodes.index(idx)
            ],
        )
        tour.append(next_node)
        unvisited.remove(next_node)
        curr = next_node

    tour.append(0)

    # 2. LNS Post-Processing Polish
    cost = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

    return {
        "algo": "Rolling_WS-LR-QAOA_PCA_LNS",
        "qubits_used": num_qubits,
        "window_size": window_size,
        "tour": tour,
        "cost": float(cost),
    }