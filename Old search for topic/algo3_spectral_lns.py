"""
Algorithm 3: Spectral Graph Clustering + Spatial LNS (Open TSP)
"""

import numpy as np


def run_algo3(data, num_qubits=5):
    matrix = data["matrix"]
    n = data["n_nodes"]

    # 1. Spectral Embedding via Graph Laplacian
    std_dev = np.std(matrix)
    scale = std_dev if std_dev > 1e-5 else 1.0
    W = np.exp(-matrix / scale)
    np.fill_diagonal(W, 0)

    D = np.diag(np.sum(W, axis=1))
    L = D - W

    evals, evecs = np.linalg.eigh(L)
    fourier_map = evecs[:, 1]  # Fiedler vector

    # Sort nodes spectrally
    sorted_nodes = list(np.argsort(fourier_map))
    if 0 in sorted_nodes:
        sorted_nodes.remove(0)

    # 2. Localized Spectral Window Traversal (Prevents long cross-cluster jumps)
    tour = [0]
    curr = 0
    unvisited = set(sorted_nodes)

    while unvisited:
        # Take the top N spectral neighbors, but pick the best spatially local stop
        window = [node for node in sorted_nodes if node in unvisited][: max(num_qubits, 5)]

        if window:
            nxt = min(window, key=lambda x: matrix[curr, x])
        else:
            nxt = min(unvisited, key=lambda x: matrix[curr, x])

        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt

    # 3. Open TSP 2-Opt Uncrossing Refinement
    improved = True
    max_iter = 150
    iter_cnt = 0

    while improved and iter_cnt < max_iter:
        improved = False
        iter_cnt += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == n - 1:
                    old_dist = matrix[tour[i - 1], tour[i]]
                    new_dist = matrix[tour[i - 1], tour[j]]
                else:
                    old_dist = matrix[tour[i - 1], tour[i]] + matrix[tour[j], tour[j + 1]]
                    new_dist = matrix[tour[i - 1], tour[j]] + matrix[tour[i], tour[j + 1]]

                if new_dist < old_dist:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
                    break
            if improved:
                break

    cost_sec = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

    return {
        "algo": "Spectral_LNS",
        "tour": tour,
        "cost": float(cost_sec),
    }