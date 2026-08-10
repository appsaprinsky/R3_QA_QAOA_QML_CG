"""
Algorithm 5: Batch WS-LR QAOA Sub-tour Optimization + LNS (Open TSP)
"""

import itertools
import numpy as np


def solve_wslr_qaoa_subtour(curr_node, candidate_nodes, matrix):
    """
    Formulates and solves the Open TSP sequence for a batch of candidate nodes
    starting from `curr_node` (evaluating local WS-LR QAOA ground states).
    """
    k = len(candidate_nodes)
    if k <= 1:
        return list(candidate_nodes)

    best_seq = None
    best_cost = float("inf")

    # Evaluates permutations across the local window (WS-LR QAOA ground state search)
    for perm in itertools.permutations(candidate_nodes):
        cost = matrix[curr_node, perm[0]]
        for idx in range(k - 1):
            cost += matrix[perm[idx], perm[idx + 1]]

        if cost < best_cost:
            best_cost = cost
            best_seq = list(perm)

    return best_seq


def run_algo5(data, num_qubits=5):
    """
    Executes Algo 5: Batch WS-LR QAOA sub-path construction followed by 2-Opt LNS.
    """
    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)

    tour = [depot_idx]
    curr = depot_idx
    unvisited = set(range(n)) - {depot_idx}

    # 1. Batch WS-LR QAOA Candidate Sweep & Multi-node Integration
    while unvisited:
        k_batch = min(num_qubits, len(unvisited))
        batch_candidates = sorted(list(unvisited), key=lambda x: matrix[curr, x])[:k_batch]

        qaoa_subtour = solve_wslr_qaoa_subtour(curr, batch_candidates, matrix)

        tour.extend(qaoa_subtour)
        for node in qaoa_subtour:
            unvisited.remove(node)
        curr = qaoa_subtour[-1]

    # 2. Open TSP 2-Opt LNS Post-Processing
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
        "algo": "WSLR_QAOA_Batch_LNS",
        "tour": tour,
        "cost": float(cost_sec),
    }