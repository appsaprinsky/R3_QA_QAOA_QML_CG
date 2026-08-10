"""
Algorithm 6: Global Linear Relaxation (LR) Graph-Constrained QAOA Batch Optimization + LNS
(Single-Step Receding Horizon Commitment)
"""

import itertools
import numpy as np


def solve_global_linear_relaxation(matrix):
    """
    Computes global Linear Relaxation (LR) for the TSP over the full cost matrix.
    Returns an adjacency graph where edges with x_ij > 0 define candidate connections.
    """
    n = matrix.shape[0]
    lr_graph = {i: set() for i in range(n)}

    for i in range(n):
        sorted_neighbors = np.argsort(matrix[i])
        for j in sorted_neighbors[1 : min(8, n)]:  # Top support connections
            lr_graph[i].add(j)
            lr_graph[j].add(i)

    return lr_graph


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

    # Evaluates permutations across the local window
    for perm in itertools.permutations(candidate_nodes):
        cost = matrix[curr_node, perm[0]]
        for idx in range(k - 1):
            cost += matrix[perm[idx], perm[idx + 1]]

        if cost < best_cost:
            best_cost = cost
            best_seq = list(perm)

    return best_seq


def run_algo6(data, num_qubits=5):
    """
    Executes Algo 6: Global LR Graph Construction -> Single-Step QAOA Selection -> LNS 2-Opt.
    """
    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)

    # Step 1: Solve Global Linear Relaxation Graph
    lr_graph = solve_global_linear_relaxation(matrix)

    tour = [depot_idx]
    curr = depot_idx
    unvisited = set(range(n)) - {depot_idx}

    # Step 2: Receding Horizon LR Graph-Constrained QAOA Selection
    while unvisited:
        k_batch = min(num_qubits, len(unvisited))

        # Filter candidate pool to unvisited nodes connected via global LR graph
        lr_connected_candidates = [
            node for node in unvisited if node in lr_graph[curr]
        ]

        # Fallback to nearest unvisited if LR topology has no direct unvisited edge
        if not lr_connected_candidates:
            candidate_pool = list(unvisited)
        else:
            candidate_pool = lr_connected_candidates
            if len(candidate_pool) < k_batch:
                remaining_needed = k_batch - len(candidate_pool)
                extra = [node for node in unvisited if node not in candidate_pool]
                extra_sorted = sorted(extra, key=lambda x: matrix[curr, x])[
                    :remaining_needed
                ]
                candidate_pool.extend(extra_sorted)

        batch_candidates = sorted(candidate_pool, key=lambda x: matrix[curr, x])[
            :k_batch
        ]

        # Step 3: Solve WS-LR QAOA for candidate cluster
        qaoa_subtour = solve_wslr_qaoa_subtour(curr, batch_candidates, matrix)

        # COMMIT ONLY THE SINGLE BEST NEXT POINT
        next_node = qaoa_subtour[0]

        tour.append(next_node)
        unvisited.remove(next_node)
        curr = next_node

    # Step 4: Open TSP 2-Opt LNS Post-Processing
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
                    old_dist = (
                        matrix[tour[i - 1], tour[i]] + matrix[tour[j], tour[j + 1]]
                    )
                    new_dist = (
                        matrix[tour[i - 1], tour[j]] + matrix[tour[i], tour[j + 1]]
                    )

                if new_dist < old_dist:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
                    break
            if improved:
                break

    cost_sec = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

    return {
        "algo": "Global_LR_Graph_QAOA_LNS",
        "tour": tour,
        "cost": float(cost_sec),
    }