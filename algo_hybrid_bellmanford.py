"""
algo_hybrid_bellmanford.py

Classical analogue of algo_hybrid_LRWSQAOA.py's run_algo_hybrid_2_5:
the SAME receding-horizon construction (select k candidates -> solve
the k-node sub-tour -> commit batch_count of it -> repeat -> 2-opt
polish), but the k-node sub-tour is solved EXACTLY via
solve_bellman_ford_subtour() (see cg_hybrid_bellmanford_sub.py's module
docstring for why that's a correct, provably-cycle-free use of
Bellman-Ford) instead of the heuristic QAOA circuit.

Why this file exists: it isolates how much of the QAOA heuristic's
result quality (or shortfall) comes from the RECEDING-HORIZON
CONSTRUCTION strategy itself (candidate selection, batching, 2-opt)
versus the QAOA solve specifically -- this file shares the former with
algo_hybrid_LRWSQAOA.py exactly, and replaces only the latter with an
exact classical solver. It's also the heuristic counterpart to
cg_hybrid_bellmanford_sub.py, so both algorithm families (receding-
horizon heuristic and column generation) have a QAOA version and a
Bellman-Ford version.

`window_size` plays the role `qubit_count` played in the QAOA version
-- renamed since there's no quantum circuit here. Same exponential-in-k
cost as cg_hybrid_bellmanford_sub.py's pricing solver (O(2^k * k^2));
window_size is capped by `bf_max_k` (default 18) for the same reason --
see that file's module docstring for measured runtimes.
"""

import math
import random
import numpy as np

from cg_hybrid_bellmanford_sub import solve_bellman_ford_subtour


def run_algo_hybrid_bf(
    data,
    window_size=5,
    exploration_percent=0.0,
    batch_count=1,
    seed=None,
    bf_max_k=18,
):
    """
    Same receding-horizon construction as run_algo_hybrid_2_5
    (algo_hybrid_LRWSQAOA.py) -- same deterministic nearest-by-distance
    + optional random exploration-slot candidate selection, same batch
    commitment, same 2-opt polish (max_iter=max(100, 50*n), matching the
    fix already applied to the QAOA version and both CG files) -- with
    the k-node sub-tour solved EXACTLY via Bellman-Ford/Held-Karp
    instead of QAOA.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)

    window_size = max(1, min(window_size, bf_max_k))
    batch_count = max(1, min(batch_count, window_size))
    exploration_percent = max(0.0, min(1.0, exploration_percent))

    tour = [depot_idx]
    curr = depot_idx
    unvisited = set(range(n)) - {depot_idx}

    while unvisited:
        k_batch = min(window_size, len(unvisited))

        if exploration_percent <= 0.0:
            n_explore = 0
            n_nearest = k_batch
        else:
            n_explore = int(math.floor(k_batch * exploration_percent))
            if k_batch > 1 and n_explore >= k_batch:
                n_explore = k_batch - 1
            n_nearest = k_batch - n_explore

        sorted_unvisited = sorted(list(unvisited), key=lambda x: (matrix[curr, x], x))
        nearest_candidates = sorted_unvisited[:n_nearest]

        remaining_unvisited = sorted_unvisited[n_nearest:]
        if n_explore > 0 and remaining_unvisited:
            exploration_candidates = random.sample(
                remaining_unvisited, min(n_explore, len(remaining_unvisited))
            )
        else:
            exploration_candidates = []

        candidate_nodes = nearest_candidates + exploration_candidates

        bf_subtour, _ = solve_bellman_ford_subtour(curr, candidate_nodes, matrix, max_k=bf_max_k)

        commit_depth = min(batch_count, len(bf_subtour))
        nodes_to_commit = bf_subtour[:commit_depth]

        tour.extend(nodes_to_commit)
        for node in nodes_to_commit:
            unvisited.remove(node)
        curr = nodes_to_commit[-1]

    # Same 2-opt fix as algo_hybrid_LRWSQAOA.py / both CG files.
    improved = True
    max_iter = max(100, 50 * n)
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
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
                    break
            if improved:
                break

    cost = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

    return {
        "algo": f"Hybrid_BF_w{window_size}_b{batch_count}_exp{int(exploration_percent*100)}",
        "tour": tour,
        "cost": float(cost),
        "params": {
            "window_size": window_size,
            "exploration_percent": exploration_percent,
            "batch_count": batch_count,
            "bf_max_k": bf_max_k,
        },
    }
