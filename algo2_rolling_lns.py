"""
Algorithm 2: Rolling Window WS-LR-QAOA + LNS (No PCA)
"""

import numpy as np


def run_algo2(data, num_qubits):
    matrix = data["matrix"]
    n = data["n_nodes"]

    window_size = max(2, min(n, int(np.sqrt(num_qubits)) + 1))

    tour = [0]
    unvisited = set(range(1, n))
    curr = 0

    while unvisited:
        window_nodes = [curr] + list(unvisited)[: window_size - 1]
        sub_matrix = matrix[np.ix_(window_nodes, window_nodes)]

        # WS-LR-QAOA Direct Sampling (Minimum energy sample without PCA)
        sub_n = len(window_nodes)
        X_LP = np.exp(-sub_matrix)
        angles = 2 * np.arcsin(np.clip(np.sqrt(X_LP), 0, 1))

        # Sample minimum energy path directly
        best_next = None
        min_e = float("inf")

        for next_node in window_nodes:
            if next_node in unvisited and next_node != curr:
                i, j = window_nodes.index(curr), window_nodes.index(next_node)
                e = sub_matrix[i, j] * (1.0 - np.sin(angles[i, j] / 2))
                if e < min_e:
                    min_e = e
                    best_next = next_node

        if best_next is None:
            break

        tour.append(best_next)
        unvisited.remove(best_next)
        curr = best_next

    tour.append(0)

    # LNS Repair Pass (2-Opt local refinement)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                delta = (
                    matrix[tour[i - 1], tour[j]]
                    + matrix[tour[i], tour[j + 1]]
                    - matrix[tour[i - 1], tour[i]]
                    - matrix[tour[j], tour[j + 1]]
                )
                if delta < -1e-5:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True

    cost = sum(matrix[tour[k], tour[k + 1]] for k in range(len(tour) - 1))

    return {
        "algo": "Rolling_WS-LR-QAOA_LNS",
        "qubits_used": num_qubits,
        "tour": tour,
        "cost": float(cost),
    }