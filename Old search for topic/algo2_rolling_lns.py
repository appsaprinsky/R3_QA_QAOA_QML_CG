"""
Algorithm 2: Spatial Rolling-Window LNS (Open TSP)
"""

import numpy as np


def run_algo2(data, num_qubits=5):
    matrix = data["matrix"]
    n = data["n_nodes"]

    tour = [0]
    curr = 0
    unvisited = set(range(1, n))

    # 1. Nearest Neighbor Window Sweep
    while unvisited:
        # Find nearest candidates to maintain local spatial coherence
        candidates = sorted(list(unvisited), key=lambda x: matrix[curr, x])[
            : max(num_qubits, 3)
        ]
        nxt = min(candidates, key=lambda x: matrix[curr, x])

        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt

    # 2. Open TSP 2-Opt Local Search
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
        "algo": "Rolling_LNS",
        "tour": tour,
        "cost": float(cost_sec),
    }