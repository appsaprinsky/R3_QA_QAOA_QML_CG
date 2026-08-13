"""
algo_hybrid_LRWSQAOA.py

Algorithm Hybrid 2+5: Spatial Rolling WS-LR QAOA with Stochastic Exploration &
Receding-Horizon Batch Commitment + LNS (Open TSP).

Contains full quantum statevector QAOA simulator for local sub-tour optimization:
1. Continuous LP Relaxation warm-start angle calculation (WS-LR).
2. QUBO -> Ising Hamiltonian mapping for positional TSP encoding.
3. Parameterized QAOA Quantum Circuit (Cost Phase + WS-X or XY Mixer Phase).
4. Variational parameter optimization (gamma, beta) via COBYLA.
5. Bitstring state measurement & constraint-repaired tour decoding.
"""

import math
import random
import numpy as np
from scipy.optimize import minimize, linprog

try:
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, Operator

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


# =====================================================================
# 1. Warm-Start QAOA Engine
# =====================================================================


def _solve_lp_relaxation(curr_node, candidates, matrix):
    """Solves continuous Linear Relaxation (LP) to extract warm-start bias weights."""
    k = len(candidates)
    n_vars = k * k  # x_{i, t} where i = candidate index, t = step index

    # Cost vector
    c = np.zeros(n_vars)
    for i in range(k):
        c[i * k + 0] = matrix[curr_node, candidates[i]]  # step 0
    for t in range(k - 1):
        for i in range(k):
            for j in range(k):
                if i != j:
                    c[i * k + t] += 0.5 * matrix[candidates[i], candidates[j]]

    # Equality constraints: sum_t x_{i,t} = 1, sum_i x_{i,t} = 1
    A_eq = []
    b_eq = []
    for i in range(k):
        row = np.zeros(n_vars)
        for t in range(k):
            row[i * k + t] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    for t in range(k):
        row = np.zeros(n_vars)
        for i in range(k):
            row[i * k + t] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    bounds = [(0.0, 1.0) for _ in range(n_vars)]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        x_lp = res.x
    else:
        x_lp = np.full(n_vars, 1.0 / k)

    eps = 1e-4
    return np.clip(x_lp, eps, 1.0 - eps)


def _build_qubo_matrix(curr_node, candidates, matrix, penalty=100.0):
    """Constructs QUBO matrix Q for Open TSP positional encoding."""
    k = len(candidates)
    n_vars = k * k
    Q = np.zeros((n_vars, n_vars))

    # 1. Cost terms
    for i in range(k):
        idx_i0 = i * k + 0
        Q[idx_i0, idx_i0] += matrix[curr_node, candidates[i]]

    for t in range(k - 1):
        for i in range(k):
            for j in range(k):
                if i != j:
                    idx_it = i * k + t
                    idx_jt1 = j * k + (t + 1)
                    Q[idx_it, idx_jt1] += matrix[candidates[i], candidates[j]]

    # 2. Constraints: sum_t x_{i,t} = 1  => penalty * (sum_t x_{i,t} - 1)^2
    for i in range(k):
        for t1 in range(k):
            idx1 = i * k + t1
            Q[idx1, idx1] += penalty * (1.0 - 2.0)
            for t2 in range(k):
                idx2 = i * k + t2
                Q[idx1, idx2] += penalty

    # 3. Constraints: sum_i x_{i,t} = 1  => penalty * (sum_i x_{i,t} - 1)^2
    for t in range(k):
        for i1 in range(k):
            idx1 = i1 * k + t
            Q[idx1, idx1] += penalty * (1.0 - 2.0)
            for i2 in range(k):
                idx2 = i2 * k + t
                Q[idx1, idx2] += penalty

    return Q


def _qubo_energy(bitstring, Q):
    """Calculates classical energy <x|Q|x> for a binary bitstring array x."""
    x = np.array(bitstring, dtype=float)
    return float(x.T @ Q @ x)


def _qaoa_statevector_simulation(
    Q, x_lp, p=1, xy_mixer=False, num_steps=30
):
    """Executes variational QAOA statevector simulation via Qiskit or fallback."""
    n_qubits = len(x_lp)
    thetas = 2.0 * np.arcsin(np.sqrt(x_lp))

    def build_circuit(params):
        gammas = params[:p]
        betas = params[p:]

        if HAS_QISKIT:
            qc = QuantumCircuit(n_qubits)
            for i in range(n_qubits):
                qc.ry(thetas[i], i)

            for layer in range(p):
                gamma = gammas[layer]
                beta = betas[layer]

                for i in range(n_qubits):
                    if Q[i, i] != 0:
                        qc.rz(2.0 * gamma * Q[i, i], i)
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        coeff = Q[i, j] + Q[j, i]
                        if coeff != 0:
                            qc.rzz(gamma * coeff, i, j)

                if xy_mixer:
                    for i in range(n_qubits - 1):
                        qc.rxx(beta, i, i + 1)
                        qc.ryy(beta, i, i + 1)
                else:
                    for i in range(n_qubits):
                        qc.ry(-thetas[i], i)
                        qc.rx(2.0 * beta, i)
                        qc.ry(thetas[i], i)

            return qc
        return None

    def objective(params):
        if HAS_QISKIT:
            qc = build_circuit(params)
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities()

            energy = 0.0
            for idx, prob in enumerate(probs):
                if prob > 1e-6:
                    bitstr = [int(b) for b in format(idx, f"0{n_qubits}b")[::-1]]
                    energy += prob * _qubo_energy(bitstr, Q)
            return energy
        return 0.0

    init_params = np.array([0.1] * p + [0.5] * p)

    if HAS_QISKIT:
        res = minimize(objective, init_params, method="COBYLA", options={"maxiter": num_steps})
        optimal_qc = build_circuit(res.x)
        sv = Statevector.from_instruction(optimal_qc)
        probs = sv.probabilities()

        best_bitstr = None
        min_e = float("inf")

        for idx, prob in enumerate(probs):
            if prob > 1e-8:
                bitstr = [int(b) for b in format(idx, f"0{n_qubits}b")[::-1]]
                e = _qubo_energy(bitstr, Q)
                if e < min_e:
                    min_e = e
                    best_bitstr = bitstr

        return best_bitstr

    return None


def solve_wslr_qaoa_subtour(
    curr_node, candidate_nodes, matrix, xy_mixer=False, p_layers=1
):
    """Formulates Open TSP QUBO, solves QAOA simulation, and decodes sequence."""
    k = len(candidate_nodes)
    if k <= 1:
        return list(candidate_nodes)

    x_lp = _solve_lp_relaxation(curr_node, candidate_nodes, matrix)
    Q = _build_qubo_matrix(curr_node, candidate_nodes, matrix)
    bitstring = _qaoa_statevector_simulation(
        Q, x_lp, p=p_layers, xy_mixer=xy_mixer
    )

    selected_tour = []
    used_candidates = set()

    for t in range(k):
        best_cand = None
        max_val = -1.0
        for i in range(k):
            if i not in used_candidates:
                val = bitstring[i * k + t] if bitstring else 0
                if val > max_val:
                    max_val = val
                    best_cand = i

        if best_cand is None:
            remaining = sorted(list(set(range(k)) - used_candidates))
            best_cand = remaining[0]

        used_candidates.add(best_cand)
        selected_tour.append(candidate_nodes[best_cand])

    return selected_tour


# =====================================================================
# 2. Main Algorithm Hybrid 2+5 Interface
# =====================================================================


def run_algo_hybrid_2_5(
    data,
    qubit_count=5,
    exploration_percent=0.0,
    batch_count=1,
    xy_mixer=False,
    seed=None,
):
    """
    Executes Hybrid Algo 2+5 with strict candidate routing.
    Guarantees zero exploration when exploration_percent == 0.0.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)

    qubit_count = max(1, qubit_count)
    batch_count = max(1, min(batch_count, qubit_count))
    exploration_percent = max(0.0, min(1.0, exploration_percent))

    tour = [depot_idx]
    curr = depot_idx
    unvisited = set(range(n)) - {depot_idx}

    while unvisited:
        k_batch = min(qubit_count, len(unvisited))

        # Strict exploration budget allocation
        if exploration_percent <= 0.0:
            n_explore = 0
            n_nearest = k_batch
        else:
            n_explore = int(math.floor(k_batch * exploration_percent))
            if k_batch > 1 and n_explore >= k_batch:
                n_explore = k_batch - 1
            n_nearest = k_batch - n_explore

        # Deterministic sorting by distance, broken by node index
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

        qaoa_subtour = solve_wslr_qaoa_subtour(
            curr, candidate_nodes, matrix, xy_mixer=xy_mixer
        )

        commit_depth = min(batch_count, len(qaoa_subtour))
        nodes_to_commit = qaoa_subtour[:commit_depth]

        tour.extend(nodes_to_commit)
        for node in nodes_to_commit:
            unvisited.remove(node)
        curr = nodes_to_commit[-1]

    # Open TSP 2-Opt Local Search (LNS)
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
        "algo": f"Hybrid_2_5_q{qubit_count}_b{batch_count}_exp{int(exploration_percent*100)}_xy{xy_mixer}",
        "tour": tour,
        "cost": float(cost_sec),
        "params": {
            "qubit_count": qubit_count,
            "exploration_percent": exploration_percent,
            "batch_count": batch_count,
            "xy_mixer": xy_mixer,
        },
    }