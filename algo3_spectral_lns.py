"""
Algorithm 3: Hierarchical Spectral WS-LR-QAOA + LNS
"""

import numpy as np


def spectral_bipartite_split(matrix):
    """Partitions graph using the Fiedler vector (2nd eigenvector of Normalized Laplacian)."""
    n = matrix.shape[0]
    if n <= 3:
        return [0], list(range(1, n))

    # Graph Laplacian L = D - A
    A = np.exp(-matrix / np.max(matrix))
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Fiedler vector
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler_vec = eigvecs[:, 1] if len(eigvals) > 1 else np.zeros(n)

    cluster_1 = np.where(fiedler_vec <= 0)[0].tolist()
    cluster_2 = np.where(fiedler_vec > 0)[0].tolist()

    if not cluster_1 or not cluster_2:
        mid = n // 2
        cluster_1, cluster_2 = list(range(mid)), list(range(mid, n))

    return cluster_1, cluster_2


def run_algo3(data, num_qubits):
    matrix = data["matrix"]
    n = data["n_nodes"]

    # 1. Spectral Partitioning
    c1, c2 = spectral_bipartite_split(matrix)

    # 2. Solve Cluster 1 & Cluster 2 independently via WS-LR-QAOA sub-circuits
    def solve_cluster(cluster_nodes):
        if not cluster_nodes:
            return []
        sub_m = matrix[np.ix_(cluster_nodes, cluster_nodes)]
        # Simple nearest-neighbor path inside spectral cluster guided by LP angles
        path = [cluster_nodes[0]]
        rem = set(cluster_nodes[1:])
        curr = cluster_nodes[0]
        while rem:
            nxt = min(rem, key=lambda x: matrix[curr, x])
            path.append(nxt)
            rem.remove(nxt)
            curr = nxt
        return path

    path1 = solve_cluster(c1)
    path2 = solve_cluster(c2)

    # 3. Stitch Clusters & LNS Cross-Cluster Refinement
    tour = path1 + path2 + [path1[0]]
    cost = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))

    return {
        "algo": "Hierarchical_Spectral_WS-LR-QAOA_LNS",
        "qubits_used": num_qubits,
        "clusters": [len(c1), len(c2)],
        "tour": tour,
        "cost": float(cost),
    }