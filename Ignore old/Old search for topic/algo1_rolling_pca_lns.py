"""
Algorithm 1: Rolling-Window PCA-Projected LNS (Open TSP)
"""

import numpy as np


def run_algo1(data, num_qubits=5):
    matrix = data["matrix"]
    coords = data["coords"]
    n = data["n_nodes"]

    # 1. Spatial PCA Projection for global directionality
    mean_coords = np.mean(coords, axis=0)
    centered_coords = coords - mean_coords
    cov_matrix = np.cov(centered_coords, rowvar=False)

    evals, evecs = np.linalg.eigh(cov_matrix)
    pc1 = evecs[:, np.argmax(evals)]
    projections = centered_coords @ pc1

    sorted_nodes = list(np.argsort(projections))
    if 0 in sorted_nodes:
        sorted_nodes.remove(0)

    # 2. Rolling Window Construction (Starting at depot 0)
    tour = [0]
    curr = 0
    unvisited = set(sorted_nodes)

    while unvisited:
        window = [
            node for node in sorted_nodes if node in unvisited
        ][: max(num_qubits, 3)]

        if window:
            nxt = min(window, key=lambda x: matrix[curr, x])
        else:
            nxt = min(unvisited, key=lambda x: matrix[curr, x])

        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt

    # 3. Open TSP 2-Opt Refinement (No depot return leg)
    improved = True
    max_iter = 100
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
        "algo": "Rolling_PCA_LNS",
        "tour": tour,
        "cost": float(cost_sec),
    }