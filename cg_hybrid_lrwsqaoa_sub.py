"""
cg_hybrid_lrwsqaoa_sub.py

Column Generation for Open TSP, using:
  - Master problem: a set-partitioning LP/ILP solved with PuLP (CBC).
  - Subproblem (pricing): the existing WS-LR QAOA sub-tour solver from
    algo_hybrid_LRWSQAOA.py (imported, not modified), used as a HEURISTIC
    column generator rather than an exact reduced-cost minimizer.

--------------------------------------------------------------------------
HOW THIS WORKS
--------------------------------------------------------------------------
Each "column" is an ordered path segment [n0, n1, ..., nk] with a cost
equal to the sum of its consecutive edge costs (an *open* sub-path cost,
no return leg). The master problem picks a set of columns whose node sets
partition {0, ..., n-1} exactly (every node covered by exactly one
column), minimizing total column cost:

    minimize   sum_p  cost(p) * x_p
    subject to sum_{p : i in p} x_p == 1   for every node i
               x_p in [0,1]  (round 1, LP relaxation)  /  {0,1} (round 2)

This is genuinely only TWO master solves, as specified:

  Round 1 (LP relaxation): solved with a trivial initial column pool
  (one singleton column per node, cost 0) purely to obtain dual prices
  pi_i for each node's covering constraint. This is not meant to be a
  good solution on its own -- singleton columns are a feasibility floor.

  Pricing (heuristic, not exact): for EVERY node i as a candidate
  segment start, take its qubit_count nearest (+ exploration) neighbors
  as candidates, run the existing solve_wslr_qaoa_subtour(i, candidates,
  matrix) to get an ordered sub-tour, and form the full column
  [i] + subtour. Rather than adding only that one path, every prefix of
  it is added as its own column -- e.g. for full column [2, 4, 9, 1]:
  [2, 4, 9, 1], [2, 4, 9], [2, 4], [2]. Each prefix's reduced cost
  (cost - sum of duals of its nodes) is computed using the Round-1
  duals, and only improving (reduced_cost < 0) columns are kept by
  default -- this mimics the spirit of CG pricing even though the QAOA
  solver isn't literally minimizing reduced cost, it's just evaluated
  after the fact. The original singleton columns are always kept too,
  as a feasibility safety net.

  Round 2 (final, integer): solve the master AGAIN, now as a binary ILP
  over the enlarged pool (initial + priced columns), to get an exact
  0/1 partition. The selected columns are then chained into a single
  Open TSP tour (the depot's column goes first; remaining columns are
  greedily chained by nearest segment-start to the current tour's last
  node) and polished with the same style of 2-opt local search used in
  algo_hybrid_LRWSQAOA.py (reimplemented here rather than imported,
  since it isn't exposed as a standalone function there and that file
  is not to be modified).

--------------------------------------------------------------------------
DEPOT HANDLING
--------------------------------------------------------------------------
The depot must end up as the tour's first node. To guarantee that
without special-casing the master problem itself, depot_idx is EXCLUDED
from every OTHER node's candidate list during pricing -- so depot can
only ever appear in a column that was priced starting from depot_idx
itself. That guarantees whichever column in the final selection contains
the depot has it as nodes[0], and that column is used to open the tour.

--------------------------------------------------------------------------
PERFORMANCE NOTE
--------------------------------------------------------------------------
"Every data point" as a pricing start means O(n) QAOA subproblem solves
for the pricing round (one per node), each on a qubit_count-sized
sub-problem -- similar order of magnitude to the original algorithm run
with batch_count=1. For real Amazon routes (100-250+ stops) this is not
free; `max_pricing_nodes` lets you subsample starting nodes instead of
using literally every one, if a full run is too slow. Default is None
(use every node), matching what was asked for.
--------------------------------------------------------------------------
"""

import math
import random
import warnings
import numpy as np

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

from algo_hybrid_LRWSQAOA import solve_wslr_qaoa_subtour


# =====================================================================
# Small shared helpers
# =====================================================================

def _open_path_cost(nodes, matrix):
    if len(nodes) < 2:
        return 0.0
    return float(sum(matrix[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)))


def _nearest_and_explore_candidates(curr_node, exclude, matrix, k, exploration_percent, rng):
    n = matrix.shape[0]
    others = [i for i in range(n) if i != curr_node and i not in exclude]
    if not others:
        return []
    others_sorted = sorted(others, key=lambda x: (matrix[curr_node, x], x))
    k = min(k, len(others_sorted))

    if exploration_percent <= 0.0 or k <= 1:
        return others_sorted[:k]

    n_explore = int(math.floor(k * exploration_percent))
    if n_explore >= k:
        n_explore = k - 1
    n_nearest = k - n_explore

    nearest = others_sorted[:n_nearest]
    remaining = others_sorted[n_nearest:]
    if n_explore > 0 and remaining:
        n_pick = min(n_explore, len(remaining))
        idx = rng.choice(len(remaining), size=n_pick, replace=False)
        explore = [remaining[i] for i in idx]
    else:
        explore = []
    return nearest + explore


def _two_opt_open_tsp(tour, matrix, max_iter=100):
    """
    Same style of open-TSP 2-opt local search used in
    algo_hybrid_LRWSQAOA.py's run_algo_hybrid_2_5 -- reimplemented here
    (not imported) since it isn't factored out as a standalone function
    in that file and that file is not to be modified for this task.
    """
    tour = list(tour)
    n = len(tour)
    if n < 4:
        return tour

    improved = True
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
    return tour


# =====================================================================
# Master problem (PuLP set-partitioning LP / ILP)
# =====================================================================

def _solve_master(columns, node_ids, relaxation, time_limit=60):
    """
    columns: list of dicts with at least {"nodes": [...], "cost": float}
    node_ids: iterable of every node index that must be covered exactly once
    relaxation: True -> continuous LP (for duals), False -> binary ILP (final)

    Returns: (status_str, selected_indices, x_values, duals_dict)
      duals_dict is {} when relaxation=False.
    """
    if not HAS_PULP:
        raise RuntimeError(
            "PuLP is not installed. cg_hybrid_lrwsqaoa_sub requires PuLP "
            "for the master problem; install it with `pip install pulp`."
        )

    prob = pulp.LpProblem("CG_Master_SetPartition", pulp.LpMinimize)
    cat = pulp.LpContinuous if relaxation else pulp.LpBinary
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=cat) for i in range(len(columns))]

    prob += pulp.lpSum(columns[i]["cost"] * x[i] for i in range(len(columns)))

    coverage_constraint_names = {}
    for node in node_ids:
        covering_vars = [x[i] for i, col in enumerate(columns) if node in col["nodes"]]
        cname = f"cover_{node}"
        if covering_vars:
            prob += pulp.lpSum(covering_vars) == 1, cname
        else:
            # Should not happen if the initial singleton pool is present,
            # but guard against a genuinely infeasible master rather than
            # letting PuLP fail silently on an empty sum.
            raise RuntimeError(
                f"Node {node} is not covered by any column in the pool -- "
                f"master problem would be infeasible. Check that the "
                f"initial singleton column set was included."
            )
        coverage_constraint_names[node] = cname

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    x_values = [pulp.value(var) for var in x]

    duals = {}
    if relaxation:
        for node, cname in coverage_constraint_names.items():
            constr = prob.constraints[cname]
            duals[node] = constr.pi if constr.pi is not None else 0.0

    selected = [i for i, v in enumerate(x_values) if v is not None and v > 0.5]
    return status, selected, x_values, duals


def _greedy_set_cover_fallback(columns, node_ids):
    """
    Used only if the final ILP master doesn't reach an optimal/feasible
    solution within time_limit (rare for these pool sizes, but CBC on an
    unfamiliar machine could still time out) -- a simple cheapest
    cost-per-newly-covered-node greedy set cover, which always terminates
    with a full partition since the singleton columns guarantee every
    single node is coverable on its own.
    """
    remaining = set(node_ids)
    selected = []
    pool = list(enumerate(columns))
    while remaining:
        best = None
        best_ratio = None
        for idx, col in pool:
            new_nodes = set(col["nodes"]) & remaining
            if not new_nodes:
                continue
            # only accept columns that don't reintroduce an already-covered node
            if set(col["nodes"]) - remaining:
                continue
            ratio = col["cost"] / max(len(new_nodes), 1)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best = (idx, col, new_nodes)
        if best is None:
            # fallback of last resort: singleton for an arbitrary remaining node
            node = next(iter(remaining))
            best = (None, {"nodes": [node], "cost": 0.0}, {node})
        idx, col, new_nodes = best
        selected.append(col)
        remaining -= new_nodes
        if idx is not None:
            pool = [(i, c) for i, c in pool if i != idx]
    return selected


# =====================================================================
# Column construction
# =====================================================================

def _build_initial_columns(n, matrix, depot_idx):
    """
    Singleton columns, one per node, used purely to seed a feasible
    Round-1 LP. Cost is each node's direct distance from the depot
    (0 for the depot's own singleton) rather than a flat 0 -- with an
    all-zero-cost initial pool the coverage matrix is the identity and
    the LP is forced into a degenerate all-zero-cost basis, which makes
    EVERY dual come out as exactly zero (pi = c_B against an identity
    basis). Zero duals mean reduced_cost = cost - 0 = cost, which is
    never negative for a real travel cost, so `only_improving_columns`
    would silently discard every QAOA-generated column no matter how
    good it is. Seeding singleton cost as distance-from-depot keeps the
    coverage matrix identity (so Round-1's primal solution is still
    forced/trivial, as intended -- it's only there for the duals) but
    now yields pi_i = distance(depot, i), a standard "naive star tour"
    baseline price. A priced column is then improving exactly when its
    real path cost undercuts the sum of visiting each of its nodes
    independently from the depot -- the right signal for whether
    grouping those nodes into one segment is actually worthwhile.
    """
    return [
        {"nodes": [i], "cost": float(matrix[depot_idx, i]) if i != depot_idx else 0.0}
        for i in range(n)
    ]


def _generate_priced_columns(
    matrix, depot_idx, duals, qubit_count, exploration_percent, xy_mixer,
    only_improving_columns, max_pricing_nodes, rng,
):
    n = matrix.shape[0]
    start_nodes = list(range(n))
    if max_pricing_nodes is not None and max_pricing_nodes < n:
        start_nodes = list(rng.choice(n, size=max_pricing_nodes, replace=False))

    priced = []
    for curr_node in start_nodes:
        exclude = {depot_idx} if curr_node != depot_idx else set()
        k_batch = min(qubit_count, n - 1 - len(exclude))
        if k_batch <= 0:
            continue

        candidates = _nearest_and_explore_candidates(
            curr_node, exclude, matrix, k_batch, exploration_percent, rng
        )
        if not candidates:
            continue

        try:
            subtour = solve_wslr_qaoa_subtour(curr_node, candidates, matrix, xy_mixer=xy_mixer)
        except RuntimeError:
            raise  # e.g. Qiskit missing -- surface this clearly, don't swallow it
        if not subtour:
            continue

        full_nodes = [curr_node] + subtour
        for L in range(len(full_nodes), 0, -1):
            seg = full_nodes[:L]
            cost = _open_path_cost(seg, matrix)
            reduced_cost = cost - sum(duals.get(node, 0.0) for node in seg)
            if only_improving_columns and reduced_cost >= -1e-9 and L > 1:
                # Still keep the length-1 (singleton) truncation regardless --
                # it's already in the initial pool anyway, so this is a no-op
                # for coverage, just skip re-adding a duplicate non-improving one.
                continue
            priced.append({"nodes": seg, "cost": cost, "reduced_cost": reduced_cost, "start": curr_node})

    return priced


def _dedupe_columns(columns):
    seen = {}
    for col in columns:
        key = tuple(col["nodes"])
        if key not in seen:
            seen[key] = col
    return list(seen.values())


# =====================================================================
# Segment concatenation into a single Open TSP tour
# =====================================================================

def _concatenate_segments(selected_columns, depot_idx, matrix):
    depot_segments = [c for c in selected_columns if c["nodes"][0] == depot_idx]
    other_segments = [c for c in selected_columns if c["nodes"][0] != depot_idx]

    if not depot_segments:
        # Shouldn't happen given the pricing exclusion rule, but fail
        # loudly rather than silently building a tour that doesn't start
        # at the depot.
        raise RuntimeError(
            "No selected column starts at depot_idx -- master solution "
            "does not yield a valid depot-anchored tour. This should be "
            "structurally impossible given the pricing exclusion rule; "
            "check that depot_idx was excluded from every other node's "
            "candidate list during pricing."
        )
    if len(depot_segments) > 1:
        warnings.warn(
            f"{len(depot_segments)} selected columns start at depot_idx; "
            f"using the cheapest one and re-queuing the rest as ordinary "
            f"segments. This suggests the pricing exclusion rule was "
            f"bypassed somewhere (e.g. via the initial singleton pool)."
        )
        depot_segments.sort(key=lambda c: c["cost"])
        other_segments = depot_segments[1:] + other_segments
        depot_segments = depot_segments[:1]

    tour = list(depot_segments[0]["nodes"])
    remaining = list(other_segments)

    while remaining:
        last_node = tour[-1]
        remaining.sort(key=lambda c: matrix[last_node, c["nodes"][0]])
        next_seg = remaining.pop(0)
        tour.extend(next_seg["nodes"])

    return tour


# =====================================================================
# Main entry point
# =====================================================================

def run_cg_hybrid_lrwsqaoa_sub(
    data,
    qubit_count=4,
    exploration_percent=0.0,
    xy_mixer=False,
    only_improving_columns=True,
    max_pricing_nodes=None,
    time_limit=60,
    seed=None,
):
    """
    Column-generation Open TSP solver: PuLP set-partitioning master problem
    + WS-LR QAOA sub-tour solver as the (heuristic) pricing subproblem.
    Exactly two master solves: an LP relaxation for duals, then a final
    binary ILP over the enlarged column pool. Returns the same shape of
    result dict as run_algo_hybrid_2_5, plus CG-specific diagnostics.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)
    qubit_count = max(1, qubit_count)
    exploration_percent = max(0.0, min(1.0, exploration_percent))

    node_ids = list(range(n))

    # --- Round 1: LP relaxation on trivial columns, to get duals ---
    initial_columns = _build_initial_columns(n, matrix, depot_idx)
    status1, _, _, duals = _solve_master(initial_columns, node_ids, relaxation=True, time_limit=time_limit)
    if status1 not in ("Optimal", "Not Solved"):
        warnings.warn(f"Round-1 LP relaxation status was '{status1}' (expected 'Optimal').")

    # --- Pricing: QAOA-generated columns, priced against Round-1 duals ---
    priced_columns = _generate_priced_columns(
        matrix, depot_idx, duals, qubit_count, exploration_percent, xy_mixer,
        only_improving_columns, max_pricing_nodes, rng,
    )

    full_pool = _dedupe_columns(initial_columns + priced_columns)

    # --- Round 2 (final): binary ILP over the enlarged pool ---
    status2, selected_idx, _, _ = _solve_master(full_pool, node_ids, relaxation=False, time_limit=time_limit)

    if status2 == "Optimal" and selected_idx:
        selected_columns = [full_pool[i] for i in selected_idx]
    else:
        warnings.warn(
            f"Round-2 ILP master status was '{status2}'; falling back to a "
            f"greedy set-cover over the same column pool to guarantee a "
            f"feasible tour."
        )
        selected_columns = _greedy_set_cover_fallback(full_pool, node_ids)

    covered = sorted(node for col in selected_columns for node in col["nodes"])
    if covered != node_ids:
        raise RuntimeError(
            f"Selected columns do not form a valid partition of all nodes "
            f"(covered {len(covered)}/{n}, or with duplicates) -- master "
            f"solution is not usable as-is."
        )

    raw_tour = _concatenate_segments(selected_columns, depot_idx, matrix)
    final_tour = _two_opt_open_tsp(raw_tour, matrix)
    final_cost = _open_path_cost(final_tour, matrix)

    return {
        "algo": f"CG_Hybrid_LRWSQAOA_Sub_q{qubit_count}_exp{int(exploration_percent*100)}_xy{xy_mixer}",
        "tour": final_tour,
        "cost": float(final_cost),
        "params": {
            "qubit_count": qubit_count,
            "exploration_percent": exploration_percent,
            "xy_mixer": xy_mixer,
            "only_improving_columns": only_improving_columns,
            "max_pricing_nodes": max_pricing_nodes,
        },
        "cg_diagnostics": {
            "round1_lp_status": status1,
            "round2_master_status": status2,
            "num_initial_columns": len(initial_columns),
            "num_priced_columns": len(priced_columns),
            "num_pool_columns_after_dedupe": len(full_pool),
            "num_segments_selected": len(selected_columns),
            "pre_2opt_cost": float(_open_path_cost(raw_tour, matrix)),
        },
    }
