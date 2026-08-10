"""
Algorithm 4: Relational & Non-Island Maker LP-Biased Quantum Annealing (Open TSP)
"""

import numpy as np


def non_island_lp_maker(matrix):
    n = matrix.shape[0]
    std_dev = np.std(matrix)
    scale = std_dev if std_dev > 1e-5 else 1.0

    X_LP = np.exp(-matrix / scale)
    np.fill_diagonal(X_LP, 0)

    min_flow = 0.2 / max(n, 1)
    X_LP = np.maximum(X_LP, min_flow)
    np.fill_diagonal(X_LP, 0)

    row_sums = X_LP.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return X_LP / row_sums


def run_algo4(data, num_qubits=5):
    matrix = data["matrix"]
    n = data["n_nodes"]

    # 1. Non-Island Continuous Backbone
    X_LP = non_island_lp_maker(matrix)

    # 2. Relational QUBO Mapping
    h_biases = matrix * (1.0 - 0.5 * X_LP)

    active_edges = []
    for i in range(n):
        for j in range(n):
            if i != j and X_LP[i, j] > (0.1 / n):
                active_edges.append((i, j, h_biases[i, j]))

    active_edges = sorted(active_edges, key=lambda x: x[2])

    edge_dict = {}
    for i, j, cost in active_edges:
        if i not in edge_dict:
            edge_dict[i] = []
        edge_dict[i].append((j, cost))

    # 3. Open TSP Tour Construction (Depot 0 start, NO depot loopback)
    visited = [0]
    curr = 0
    unvisited = set(range(1, n))

    while unvisited:
        candidates = [
            (j, cost) for (j, cost) in edge_dict.get(curr, []) if j in unvisited
        ]

        if candidates:
            nxt = min(candidates, key=lambda x: x[1])[0]
        else:
            nxt = min(unvisited, key=lambda x: matrix[curr, x])

        visited.append(nxt)
        unvisited.remove(nxt)
        curr = nxt

    tour = visited

    # Calculate exact route duration (excluding return-to-depot stem)
    cost_sec = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

    return {
        "algo": "Relational_NonIsland_QuantumAnnealing",
        "qubits_used": num_qubits,
        "tour": tour,
        "cost": float(cost_sec),
    }