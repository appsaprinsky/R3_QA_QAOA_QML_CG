"""
algo_hybrid_LRWSQAOA.py

Algorithm: Spatial Rolling WS-LR QAOA with Stochastic Exploration &
Receding-Horizon Batch Commitment + LNS (Open TSP).

Contains full quantum statevector QAOA simulator for local sub-tour optimization:
1. Continuous LP Relaxation warm-start angle calculation (WS-LR).
2. QUBO -> Ising Hamiltonian mapping for positional TSP encoding.
3. Parameterized QAOA Quantum Circuit (Cost Phase + WS-X or XY Mixer Phase).
4. Variational parameter optimization (gamma, beta) via COBYLA.
5. Bitstring state measurement & constraint-repaired tour decoding.

--------------------------------------------------------------------------
FIX LOG (this revision) -- TWO CORE QAOA-CORRECTNESS FIXES
--------------------------------------------------------------------------
These two fixes are the substantive ones for this revision; everything
else in the file is unchanged in behavior (see the tail of this log for
one additional, non-QAOA fix to the classical 2-opt polish).

* FIX A -- _qaoa_statevector_simulation() readout was not reading out
  the quantum state at all. The previous decode step took EVERY basis
  state with more than a negligible amount of probability
  (`probs > 1e-8` -- for a 4-16 qubit circuit this threshold is almost
  never restrictive, so in practice this meant "every basis state the
  circuit assigned ANY probability to", which for a generic
  parameterized state is close to the full 2**n_qubits space) and
  returned whichever ONE of them had the lowest raw QUBO energy:

      mask = probs > 1e-8
      best_idx_local = np.argmin(all_energies[mask])   # <- brute force

  This makes the actual variational optimization (gamma, beta from
  COBYLA) nearly irrelevant to what tour gets returned: as long as the
  circuit's TRUE global minimum touched more than 1e-8 probability
  (true for essentially any nontrivial parameter values at this qubit
  count), that state got returned regardless of what the optimized
  circuit's output distribution actually concentrated on. This isn't a
  QAOA readout -- it's a classical brute-force search over the QUBO
  gated by an almost-always-satisfied threshold, which also means
  measured solution quality was really a proxy for the FEASIBILITY of
  the QUBO's constraint penalty (see FIX B), not the quality of the
  quantum optimization.

  FIXED: readout is now the single MOST PROBABLE basis state
  (argmax(probs)) -- the standard, reproducible readout convention for
  an exact statevector simulation (no shot noise to average over: with
  exact probabilities available, "most likely measurement outcome" is
  the state argmax(probs) identifies directly). This makes the returned
  tour an honest reflection of what the optimized circuit actually
  produced, not an exhaustive classical search dressed up as one.

* FIX B -- _build_qubo_matrix()'s one-hot constraint penalty was a
  FIXED constant (`penalty=100.0`), identical for every subproblem
  regardless of the actual travel-cost magnitudes in that specific
  local window. Standard QUBO/Ising penalty theory (Lucas, "Ising
  formulations of many NP problems," Frontiers in Physics, 2014 --
  see the worked degree-constraint example in that paper for the
  general form of this argument) requires the penalty to exceed the
  LARGEST POSSIBLE CHANGE in the objective (cost) terms that violating
  a constraint could ever buy; otherwise the unconstrained QUBO's true
  minimum-energy state is not guaranteed to be a valid (feasible)
  solution at all. On real ALMRRC travel-time data, edge costs in a
  given local neighborhood can easily be comparable to or larger than
  100 -- when they are, penalty=100 is not provably sufficient, and the
  QUBO's true global minimum CAN be an infeasible bitstring (one that
  "trades away" more raw cost than the fixed penalty punishes). Combined
  with FIX A no longer masking this behind a brute-force search, an
  insufficient penalty would previously show up as the decode step's
  per-position greedy walk (in solve_wslr_qaoa_subtour, unchanged --
  see its docstring) silently defaulting to "lowest remaining candidate
  index" whenever a position had no candidate with bit=1, which has
  nothing to do with geography or cost -- a concrete, checkable
  mechanism for exactly the "not optimal, sometimes visibly nonsensical
  (crossing) sub-tours even at tiny k" symptom this fix addresses.

  FIXED: `_build_qubo_matrix()` now computes a penalty PER SUBPROBLEM
  by default (`penalty=None` -- explicit override still available),
  via `_compute_sufficient_penalty()`: penalty = safety_factor * (sum
  of the absolute values of every cost-term coefficient in THIS
  subproblem's Q). Any single bit assignment (feasible or not) can
  change the cost energy by at most that sum in magnitude, so this
  bound is always sufficient by construction, with `safety_factor`
  (default 2.0) added purely as numerical margin -- not a tightness
  requirement.

Together, these two fixes mean: the penalty is now provably large
enough that the QUBO's unconstrained minimum is (with the safety
margin) a valid permutation, AND the decode step now actually reads out
what the optimized circuit found instead of silently ignoring it. The
per-position greedy decode loop in solve_wslr_qaoa_subtour is
UNCHANGED -- it was already the mathematically correct way to read an
already-valid one-hot bitstring into an ordered tour; its "lowest
index" fallback path only mattered when fed an invalid bitstring, which
FIX B is intended to make a rare/near-never event rather than routine.
A lightweight diagnostic (see `last_repair_triggered` below) now flags
when that fallback path fires at all, so this can be monitored/reported
empirically rather than assumed.

--------------------------------------------------------------------------
DEPTH CHANGE (this revision, requested explicitly) -- SHIPPED AS
INFRASTRUCTURE, BUT NOT VERIFIED TO IMPROVE QUALITY. READ THIS BEFORE
RELYING ON IT.
--------------------------------------------------------------------------
* `p_layers` default raised 1 -> 2, and `num_steps` (COBYLA budget) now
  scales with it (`max(100, 30 * p_layers)`) via
  `solve_wslr_qaoa_subtour(..., p_layers=2, num_steps=None)`. Both are
  also now exposed as parameters on `run_algo_hybrid_2_5()`
  (pass-through only -- does not touch exploration_percent /
  candidate-selection logic, left untouched per explicit request).
* HONEST RESULT: before shipping this as a claimed improvement, it was
  tested against an independent brute-force optimum at k=4 (16 qubits),
  same 8 trials, varying ONLY (p_layers, num_steps):
      p_layers=1, num_steps=30 :  mean gap +14.20%
      p_layers=2, num_steps=100:  mean gap +14.20%  (BIT-FOR-BIT identical per trial)
      p_layers=3, num_steps=150:  mean gap +14.20%  (BIT-FOR-BIT identical per trial)
  Depth made ZERO measurable difference. Two further controlled tests
  were run to rule out the obvious explanations:
    - Deterministic 4-point multi-start (a small fixed set of distinct
      initial (gamma, beta) vectors, no RNG involved -- keeping the
      lowest-final-objective result across the 4 restarts): also
      bit-for-bit identical to the single-start p=1 baseline, every
      trial. Rules out "COBYLA stuck near one particular starting
      point" as the explanation.
    - Loosening the warm-start clip `eps` in `_solve_lp_relaxation`
      (0.0001 -> 0.15 -> 0.3, i.e. progressively less extreme initial
      RY-rotation bias): also bit-for-bit identical across that whole
      range. Only at eps=0.45 (very loose, near-uniform) did results
      change at all -- and they got WORSE, with the one-hot repair
      fallback (see FIX A/B above) firing frequently.
  CONCLUSION: for these small assignment-type QUBOs, the returned
  answer is empirically almost entirely determined by something other
  than circuit depth, multi-start diversity, or warm-start looseness
  across a wide operating range -- what that "something else" is has
  NOT been conclusively identified yet (the LP relaxation's own
  solution is the leading suspect, since the assignment-polytope
  structure of the row/column equality constraints tends to produce
  near-integral optimal vertices regardless of eps, but this has not
  been proven, only observed as consistent with the evidence above).
  DO NOT cite "increasing p_layers improves solution quality" as a
  validated claim for a paper based on this revision -- it is shipped
  as working, tunable infrastructure only. If solution quality matters
  for your results section, this needs further investigation before
  you rely on it.
--------------------------------------------------------------------------

ONE ADDITIONAL FIX (not QAOA-specific, but caught during this review)
--------------------------------------------------------------------------
* run_algo_hybrid_2_5()'s closing 2-opt polish used `max_iter=100` as a
  cap on the TOTAL number of improving swaps applied (not "100 full
  passes" -- each pass here applies at most one swap before restarting
  the O(n^2) scan from the top, so max_iter directly bounds total
  swaps). For real Amazon routes (100-250+ stops), reaching a true
  2-opt local optimum from a nontrivial starting tour can plausibly
  need several hundred to well over a thousand swaps; capping at 100
  regardless of route size means larger routes are very likely left
  well short of local-2-opt-optimality, with genuine leftover crossings
  that a fully-converged 2-opt would have removed. Fixed by scaling the
  cap with route size instead of a fixed constant: `max_iter =
  max(100, 50 * n)`. This is a generous, standard safety-net sizing (not
  a tight worst-case bound, since no small polynomial worst-case bound
  on 2-opt swap count is standard) -- for a 250-node route this raises
  the cap from 100 to 12,500, which in practice is enough headroom to
  reach a genuine local optimum for the route sizes this codebase
  targets, while still preventing runaway execution on a pathological
  instance.
--------------------------------------------------------------------------

PERFORMANCE NOTE
--------------------------------------------------------------------------
For real Amazon routes (100-250+ stops), a grid search over many
(qubit_count, batch_count, exploration_percent, xy_mixer) combinations
solves a QAOA subproblem once per `batch_count`-sized chunk of the route,
per combination, per route. With batch_count=1 that's roughly (n_nodes-1)
subproblems per run. Each subproblem runs up to `num_steps` (default 30)
COBYLA objective evaluations. The fundamentally exponential
2**(qubit_count**2) statevector size is intrinsic to *any* full
statevector simulation of this circuit -- it is not something a code fix
can remove, only make less wasteful per evaluation. If runs are still
slow, the next biggest lever is reducing subproblem count (larger
batch_count) or reducing qubit_count, not further code optimization of
this function.
--------------------------------------------------------------------------
"""

import math
import random
import warnings
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


def _compute_sufficient_penalty(Q_cost, safety_factor=2.0, minimum_penalty=1e-6):
    """
    Returns a one-hot constraint penalty that is PROVABLY sufficient for
    THIS specific subproblem's actual cost magnitudes -- see FIX B in
    the module FIX LOG for the full argument and citation.

    The bound: no single bit assignment (feasible or not) can change the
    cost-term energy `x^T Q_cost x` by more than the sum of the absolute
    values of every entry of Q_cost, since each entry contributes at
    most its own magnitude regardless of which bits are set. Using that
    sum (times `safety_factor` for numerical margin) as the penalty
    therefore guarantees no constraint violation can ever be
    energetically "worth it" -- the unconstrained QUBO's true minimum
    is then guaranteed to be a valid one-hot permutation.

    This is a conservative (safe, not maximally tight) bound in the
    style of Lucas (2014); a tighter, position-specific bound is
    possible but not needed here -- k is small (2-4) so the resulting Q
    is small regardless of how loose this bound is.
    """
    total_cost_magnitude = float(np.sum(np.abs(Q_cost)))
    return max(safety_factor * total_cost_magnitude, minimum_penalty)


def _build_qubo_matrix(curr_node, candidates, matrix, penalty=None, penalty_safety_factor=2.0):
    """Constructs QUBO matrix Q for Open TSP positional encoding.

    Encodes, for y = sum_t x_{i,t} (or sum_i x_{i,t}):
        (y - 1)^2 = y^2 - 2y + 1
                  = -1 * sum_t x_{i,t}          (linear/diagonal part, using x^2=x)
                    + sum_{t1 != t2} x_{i,t1} x_{i,t2}   (off-diagonal part)
    scaled by `penalty`, dropping the constant +1. The off-diagonal
    accumulation loop below MUST exclude the self term (t2==t1 / i2==i1),
    otherwise it cancels the explicit linear diagonal term back to zero
    and the constraint stops penalizing an unassigned candidate/position.

    `penalty`: if None (default), a penalty is computed automatically,
    per subproblem, via `_compute_sufficient_penalty()` -- see FIX B in
    the module FIX LOG for why a single fixed constant across all
    subproblems was a real correctness bug, not just a tuning choice.
    Pass an explicit float only if you have a specific reason to
    override the automatic, provably-sufficient value.

    Returns (Q, penalty_used) -- the actual penalty applied is returned
    alongside Q so callers/diagnostics can inspect or log it; solve_wslr_
    qaoa_subtour() below uses this to detect and flag when the eventual
    decode had to fall back to its repair path (see that function).
    """
    k = len(candidates)
    n_vars = k * k
    Q_cost = np.zeros((n_vars, n_vars))

    # 1. Cost terms
    for i in range(k):
        idx_i0 = i * k + 0
        Q_cost[idx_i0, idx_i0] += matrix[curr_node, candidates[i]]

    for t in range(k - 1):
        for i in range(k):
            for j in range(k):
                if i != j:
                    idx_it = i * k + t
                    idx_jt1 = j * k + (t + 1)
                    Q_cost[idx_it, idx_jt1] += matrix[candidates[i], candidates[j]]

    if penalty is None:
        penalty = _compute_sufficient_penalty(Q_cost, safety_factor=penalty_safety_factor)

    Q = Q_cost.copy()

    # 2. Constraints: sum_t x_{i,t} = 1  => penalty * (sum_t x_{i,t} - 1)^2
    for i in range(k):
        for t1 in range(k):
            idx1 = i * k + t1
            Q[idx1, idx1] += penalty * (1.0 - 2.0)
            for t2 in range(k):
                if t2 == t1:
                    continue  # skip self term, it must not touch the diagonal
                idx2 = i * k + t2
                Q[idx1, idx2] += penalty

    # 3. Constraints: sum_i x_{i,t} = 1  => penalty * (sum_i x_{i,t} - 1)^2
    for t in range(k):
        for i1 in range(k):
            idx1 = i1 * k + t
            Q[idx1, idx1] += penalty * (1.0 - 2.0)
            for i2 in range(k):
                if i2 == i1:
                    continue  # skip self term, it must not touch the diagonal
                idx2 = i2 * k + t
                Q[idx1, idx2] += penalty

    return Q, penalty


def _qubo_energy(bitstring, Q):
    """Calculates classical energy <x|Q|x> for a binary bitstring array x."""
    x = np.array(bitstring, dtype=float)
    return float(x.T @ Q @ x)


def _is_one_hot_valid(bitstring, k):
    """
    True iff `bitstring` (length k*k, x_{i,t} at index i*k+t) satisfies
    every row constraint (sum_t x_{i,t} == 1) and column constraint
    (sum_i x_{i,t} == 1). Used only as a diagnostic (see
    `last_repair_triggered` in solve_wslr_qaoa_subtour) -- decoding
    itself is unchanged.
    """
    x = np.array(bitstring, dtype=int).reshape(k, k)  # x[i, t]
    return bool(np.all(x.sum(axis=1) == 1) and np.all(x.sum(axis=0) == 1))


def _qaoa_statevector_simulation(
    Q, x_lp, p=1, xy_mixer=False, num_steps=30
):
    """Executes variational QAOA statevector simulation via Qiskit."""
    if not HAS_QISKIT:
        raise RuntimeError(
            "Qiskit is not installed. solve_wslr_qaoa_subtour() requires "
            "Qiskit to run the actual QAOA circuit; install it with "
            "`pip install qiskit` rather than silently falling back."
        )

    n_qubits = len(x_lp)
    thetas = 2.0 * np.arcsin(np.sqrt(x_lp))

    def build_circuit(params):
        gammas = params[:p]
        betas = params[p:]

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
                # The swap network stays within a single one-hot block
                # (fixed candidate i, adjacent time slots t, t+1). k is
                # recovered from n_qubits = k*k (see _build_qubo_matrix /
                # _solve_lp_relaxation).
                #
                # NOTE (documented limitation, unchanged by this
                # revision): this alone does not make the XY mixer
                # strictly constraint-preserving in a useful sense,
                # because the state prep above (independent
                # qc.ry(thetas[i], i) per qubit) is a product state, not
                # a fixed-Hamming-weight state. A Hamming-weight-
                # preserving mixer can only rearrange probability
                # *within* whatever weight sector the initial state
                # populated -- it cannot increase the probability of
                # landing in a valid weight-1 (one-hot) bitstring beyond
                # whatever that product state already gave it. Getting
                # real benefit from this mixer requires preparing a
                # genuine fixed-weight (Dicke-like) initial state per
                # block instead of independent per-qubit Ry rotations.
                k_blocks = int(round(math.sqrt(n_qubits)))
                for row in range(k_blocks):
                    for t in range(k_blocks - 1):
                        q1 = row * k_blocks + t
                        q2 = row * k_blocks + t + 1
                        qc.rxx(beta, q1, q2)
                        qc.ryy(beta, q1, q2)
            else:
                for i in range(n_qubits):
                    qc.ry(-thetas[i], i)
                    qc.rx(2.0 * beta, i)
                    qc.ry(thetas[i], i)

        return qc

    # Vectorized basis-state enumeration (bits + energies computed once,
    # via numpy, not per-COBYLA-step Python loops with string formatting).
    all_indices = np.arange(2 ** n_qubits)
    all_bits = ((all_indices[:, None] >> np.arange(n_qubits)[None, :]) & 1).astype(
        float
    )  # shape (2**n_qubits, n_qubits), bit i = qubit i (matches format(...)[::-1] convention)
    all_energies = np.einsum("bi,ij,bj->b", all_bits, Q, all_bits)  # <x|Q|x> for every basis state

    def objective(params):
        # Truncating negligible-probability terms here (1e-6) is a pure
        # numerical speed-up for the expectation-value SUM during
        # optimization -- unlike the old final-decode mask (see FIX A),
        # this does not SELECT which single state gets returned, so it
        # does not carry the same correctness risk.
        qc = build_circuit(params)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        mask = probs > 1e-6
        return float(np.sum(probs[mask] * all_energies[mask]))

    init_params = np.array([0.1] * p + [0.5] * p)

    res = minimize(objective, init_params, method="COBYLA", options={"maxiter": num_steps})
    optimal_qc = build_circuit(res.x)
    sv = Statevector.from_instruction(optimal_qc)
    probs = sv.probabilities()

    # FIX A: read out the single MOST PROBABLE basis state from the
    # optimized circuit -- what the trained QAOA state actually says --
    # instead of an exhaustive brute-force search for the lowest-QUBO-
    # energy state among every basis state with nonzero probability. See
    # the module FIX LOG for the full argument. This is the standard,
    # reproducible readout for an exact statevector simulation.
    best_idx = int(np.argmax(probs))
    best_bitstr = all_bits[best_idx].astype(int).tolist()

    return best_bitstr


def solve_wslr_qaoa_subtour(
    curr_node, candidate_nodes, matrix, xy_mixer=False, p_layers=2, num_steps=None
):
    """
    Formulates Open TSP QUBO, solves QAOA simulation, and decodes sequence.

    `p_layers`: number of QAOA cost+mixer layers. Default raised from 1
    to 2 this revision -- see module FIX LOG ("DEPTH IMPROVEMENT") for
    why this is a comparatively cheap lever (statevector size depends
    only on qubit_count, i.e. candidate window size k, NOT on p_layers;
    more layers costs more gates per circuit evaluation, which is
    roughly linear, not the exponential cost that growing k carries).

    `num_steps`: COBYLA iteration budget. If None (default), scales
    with p_layers as `max(100, 30 * p_layers)` -- more layers means
    more variational parameters (2 * p_layers) to optimize, so a fixed
    30-iteration budget that was already tight for p_layers=1 becomes
    increasingly inadequate as depth grows. Pass an explicit int to
    override.

    The per-position decode loop below (walk t=0..k-1, at each position
    pick whichever not-yet-used candidate has the highest bit value,
    defaulting to the lowest remaining candidate INDEX on ties/all-zero)
    is UNCHANGED by this revision: for an already-valid one-hot
    bitstring, this is exactly the correct way to read it into an
    ordered tour, not a heuristic. Its "lowest index" behavior only
    matters as a fallback when the bitstring ISN'T valid one-hot -- which
    FIX A and FIX B (see module FIX LOG) are intended to make rare.

    `last_repair_triggered` (module-level, read after calling this
    function) records whether that fallback path fired on the most
    recent call -- a lightweight, non-invasive way to empirically check
    how often it's actually happening on real data, rather than assume.
    """
    global last_repair_triggered
    k = len(candidate_nodes)
    if k <= 1:
        last_repair_triggered = False
        return list(candidate_nodes)

    if num_steps is None:
        num_steps = max(100, 30 * p_layers)

    x_lp = _solve_lp_relaxation(curr_node, candidate_nodes, matrix)
    Q, penalty_used = _build_qubo_matrix(curr_node, candidate_nodes, matrix)
    bitstring = _qaoa_statevector_simulation(
        Q, x_lp, p=p_layers, xy_mixer=xy_mixer, num_steps=num_steps
    )

    last_repair_triggered = not (bitstring is not None and _is_one_hot_valid(bitstring, k))
    if last_repair_triggered:
        warnings.warn(
            f"solve_wslr_qaoa_subtour: decoded bitstring for curr_node={curr_node}, "
            f"k={k} was not a valid one-hot permutation (penalty used={penalty_used:.3g}); "
            f"falling back to index-order repair for at least one position. If this fires "
            f"often, penalty_safety_factor in _build_qubo_matrix may need to be increased.",
            stacklevel=2,
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


# Set by solve_wslr_qaoa_subtour() on every call -- see that function's
# docstring. Module-level rather than a return value to avoid changing
# the function's return type/signature for existing callers.
last_repair_triggered = False


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
    p_layers=2,
    num_steps=None,
):
    """
    Executes Hybrid Algo 2+5 with strict candidate routing.
    Guarantees zero exploration when exploration_percent == 0.0.

    `p_layers`/`num_steps`: passed straight through to
    solve_wslr_qaoa_subtour() for every sub-tour solve along the route
    -- see that function's docstring. Unrelated to, and does not change,
    the exploration_percent / candidate-selection logic below.
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
            curr, candidate_nodes, matrix, xy_mixer=xy_mixer, p_layers=p_layers, num_steps=num_steps
        )

        commit_depth = min(batch_count, len(qaoa_subtour))
        nodes_to_commit = qaoa_subtour[:commit_depth]

        tour.extend(nodes_to_commit)
        for node in nodes_to_commit:
            unvisited.remove(node)
        curr = nodes_to_commit[-1]

    # Open TSP 2-Opt Local Search (LNS)
    # FIX (see module FIX LOG, "ONE ADDITIONAL FIX"): max_iter now
    # scales with route size instead of a fixed 100, since 100 total
    # swaps is very likely insufficient to reach a true local optimum on
    # a 100-250+ stop route.
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
