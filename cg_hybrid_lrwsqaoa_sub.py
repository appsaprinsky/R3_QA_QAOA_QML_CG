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
               x_p in [0,1]  (pricing iterations, LP relaxation)
               x_p in {0,1}  (final round)

--------------------------------------------------------------------------
ITERATION_CG
--------------------------------------------------------------------------
The number of pricing iterations is controlled by the module-level
constant ITERATION_CG (default 10, overridable per-call via
`n_iterations`). Each iteration:

  1. Solves the master as an LP relaxation over the CURRENT column pool
     to get fresh duals pi_i.
  2. Prices new columns against those duals: for EVERY node i as a
     candidate segment start, take its qubit_count nearest (+
     exploration) neighbors, run the existing
     solve_wslr_qaoa_subtour(i, candidates, matrix) to get an ordered
     sub-tour, and form the full column [i] + subtour. Every prefix of
     it is priced as its own column -- e.g. for full column [2, 4, 9, 1]:
     [2, 4, 9, 1], [2, 4, 9], [2, 4], [2] -- each with reduced cost
     (cost - sum of duals of its nodes) computed against THIS
     iteration's duals. Only improving (reduced_cost < 0) columns are
     kept by default (`only_improving_columns=True`); the length-1
     singleton is always kept regardless, as a feasibility safety net.
  3. Newly-priced columns are deduplicated against the existing pool
     (by node sequence) and merged in -- the pool only grows, columns
     are never removed, exactly as in standard column generation.

  Note that with exploration_percent=0 the QAOA call itself is fully
  deterministic (same candidates -> same sub-tour every time), so
  re-running iterations doesn't rediscover new PATHS on its own -- what
  changes each iteration is which of those already-known truncations
  clear the reduced-cost bar, since the duals themselves evolve as the
  pool grows and the LP relaxation's chosen basis shifts. Set
  exploration_percent > 0 to also get genuinely new candidate sets
  across iterations (more diverse columns, not just a moving filter
  over the same fixed candidates).

  CANDIDATE SELECTION (from iteration >= 2): candidate selection for the
  "nearest" slice is plain closest-by-distance on iteration 1 (duals are
  still just the naive distance-from-depot seed there, not informative
  enough to use yet -- see _build_initial_columns). From iteration 2
  onward, once duals reflect real priced structure:
    - Every candidate in the pool (subject to RADIUS_POOL_SEARCH, see
      below) gets a reduced cost computed:
      matrix[curr_node, x] - duals.get(x, 0).
    - The "nearest" slice takes the qubit_count candidates with the
      SMALLEST reduced cost (ranked ascending, not filtered by sign --
      unlike an earlier version of this file, a candidate is not
      required to have negative reduced cost to be selected, just to be
      among the best available). This guarantees a full qubit_count-sized
      "nearest" set whenever the pool has enough points, instead of
      shrinking whenever few candidates happen to be strictly negative.
    - The QAOA subproblem itself also incorporates duals from iteration
      2 onward: instead of solving over the raw distance matrix, it
      solves over a dual-adjusted matrix (matrix[i, j] - duals[j] for
      every j), so QAOA is searching for low REDUCED-cost orderings of
      the chosen candidates, not merely short ones. This only affects
      the QAOA call itself -- every cost bookkeeping computation
      (column cost, reduced cost filtering, final tour cost) continues
      to use the raw, unadjusted matrix.
  Every node gets its own independent candidate search each iteration --
  there is no shared "used" or "unvisited" set depleting across
  different starting nodes within an iteration. All n nodes are always
  pricing starts (see _generate_priced_columns's start_nodes), and every
  one of them searches the (nearly) full node universe on its own terms.

  RADIUS_POOL_SEARCH (module constant, default 1.0): restricts the
  candidate pool used for the reduced-cost ranking above to nodes within
  RADIUS_POOL_SEARCH * (largest pairwise distance in the whole matrix)
  of curr_node. At the default of 1.0 this is a no-op -- no pairwise
  distance can exceed the matrix's own global maximum, so every node is
  always in range and the full point set is searched, as it always has
  been. Lowering it restricts the search to a smaller spatial
  neighborhood, which is a lever for large instances, not something
  currently changing behavior at the default.

  Iterations stop early if a pass adds zero new columns to the pool
  (converged -- further iterations would be identical) or once
  `n_iterations` is reached, whichever comes first.

  Final round (integer): solve the master ONE more time, now as a
  binary ILP over the final pool, to get an exact 0/1 partition. The
  selected columns are then chained into a single Open TSP tour (the
  depot's column goes first; remaining columns are greedily chained by
  nearest segment-start to the current tour's last node) and polished
  with the same style of 2-opt local search used in
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
per iteration (one per node), each on a qubit_count-sized sub-problem --
similar order of magnitude to the original algorithm run with
batch_count=1. With ITERATION_CG pricing iterations, total QAOA calls
scale as O(ITERATION_CG * n) -- at the default of 10 iterations, that's
roughly 10x the pricing cost of the original 2-round version. For real
Amazon routes (100-250+ stops) this is not free; `max_pricing_nodes`
lets you subsample starting nodes per iteration instead of using
literally every one, and lowering ITERATION_CG is the other direct
lever if a run is too slow.
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

# Number of pricing iterations (LP relaxation for duals -> price -> grow
# pool) run before the single final integer master solve. Override
# per-call via run_cg_hybrid_lrwsqaoa_sub(..., n_iterations=...); this is
# just the default, and the single source of truth other modules
# (run_CG_experiment.py, visualize_step_by_step_CG.py) import rather than
# redefining locally.
ITERATION_CG = 10

# Restricts the candidate pool used for reduced-cost ranking (see
# _dual_aware_nearest_and_explore) to nodes within
# RADIUS_POOL_SEARCH * (largest pairwise distance in the whole matrix)
# of the current search point. Default 1.0 = no restriction, since no
# pairwise distance can exceed the matrix's own global maximum -- every
# node is always in range, matching "use all points" behavior. Lower
# values restrict the search to a smaller spatial neighborhood; this is
# a lever for large instances, not something changing behavior today.
RADIUS_POOL_SEARCH = 1.0


# =====================================================================
# Small shared helpers
# =====================================================================

def _open_path_cost(nodes, matrix):
    if len(nodes) < 2:
        return 0.0
    return float(sum(matrix[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)))


def _dual_aware_nearest_and_explore(curr_node, exclude, matrix, k, exploration_percent, rng,
                                     duals=None, global_max_dist=None):
    """
    Selects up to k candidates for curr_node's QAOA pricing subproblem,
    returned as (nearest, explore). Every call searches this curr_node's
    OWN candidate pool independently -- there is no shared "used" state
    across different starting nodes, so every node's search always sees
    the (nearly) full point set, never a pool depleted by some other
    node's earlier selection. This is a two-phase search:

      Phase 0 (duals is None -- iteration 1 / "iteration 0"): nearest is
      plain closest-by-distance over ALL other nodes in the graph.
      Iteration 1's duals are just the naive distance-from-depot seed
      (see _build_initial_columns), not yet meaningful enough to use.

      Phase 1+ (duals provided -- iteration >= 2 / "iteration 1" in the
      zero-indexed sense): every node in the candidate pool (optionally
      narrowed by RADIUS_POOL_SEARCH, see below) gets a reduced cost
      matrix[curr_node, x] - duals.get(x, 0). Candidates are RANKED by
      this value ascending and the k smallest are taken as "nearest" --
      this is a ranking, not a sign filter. A candidate does not need
      negative reduced cost to be selected, only to be among the best
      available. This guarantees a full k-sized "nearest" set whenever
      the pool has at least k points, rather than shrinking whenever few
      candidates happen to be strictly negative.

    RADIUS_POOL_SEARCH (module constant): when duals is provided and
    global_max_dist is given, the pool considered for ranking is first
    narrowed to nodes with matrix[curr_node, x] <= RADIUS_POOL_SEARCH *
    global_max_dist. At the default RADIUS_POOL_SEARCH=1.0 this never
    excludes anyone (no pairwise distance can exceed the matrix's own
    global maximum), so today this is a no-op and the full point set is
    always searched.

    exploration_percent alone controls how many exploration slots exist;
    this function never inflates that count to compensate for anything.
    """
    n = matrix.shape[0]
    others = [i for i in range(n) if i != curr_node and i not in exclude]
    if not others:
        return [], []
    others_sorted = sorted(others, key=lambda x: (matrix[curr_node, x], x))
    k = min(k, len(others_sorted))

    if exploration_percent <= 0.0 or k <= 1:
        n_nearest, n_explore = k, 0
    else:
        n_explore = int(math.floor(k * exploration_percent))
        if n_explore >= k:
            n_explore = k - 1
        n_nearest = k - n_explore

    if duals is not None:
        if global_max_dist is not None and global_max_dist > 0:
            radius = RADIUS_POOL_SEARCH * global_max_dist
            pool = [x for x in others if matrix[curr_node, x] <= radius]
            if not pool:
                pool = others  # radius somehow excluded everyone -- don't strand the search
        else:
            pool = others
        ranked = sorted(pool, key=lambda x: (matrix[curr_node, x] - duals.get(x, 0.0), matrix[curr_node, x], x))
        nearest = ranked[:n_nearest]
    else:
        nearest = others_sorted[:n_nearest]

    remaining = [x for x in others_sorted if x not in nearest]
    if n_explore > 0 and remaining:
        n_pick = min(n_explore, len(remaining))
        idx = rng.choice(len(remaining), size=n_pick, replace=False)
        explore = [remaining[i] for i in idx]
    else:
        explore = []
    return nearest, explore


def _nearest_and_explore_candidates(curr_node, exclude, matrix, k, exploration_percent, rng,
                                     duals=None, global_max_dist=None):
    nearest, explore = _dual_aware_nearest_and_explore(
        curr_node, exclude, matrix, k, exploration_percent, rng, duals, global_max_dist
    )
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
    only_improving_columns, max_pricing_nodes, rng, apply_dual_candidate_filter=False,
    global_max_dist=None,
):
    n = matrix.shape[0]
    start_nodes = list(range(n))
    if max_pricing_nodes is not None and max_pricing_nodes < n:
        start_nodes = list(rng.choice(n, size=max_pricing_nodes, replace=False))

    candidate_duals = duals if apply_dual_candidate_filter else None

    # QAOA itself incorporates duals from iteration >= 2 onward, matching
    # when candidate selection starts using them: the matrix handed to
    # solve_wslr_qaoa_subtour has every column j reduced by duals[j], so
    # QAOA searches for low REDUCED-cost orderings of the given
    # candidates, not merely geometrically short ones. This is the only
    # place duals affect the matrix -- every cost bookkeeping computation
    # below (column cost, reduced cost, dedup, master objective) keeps
    # using the raw, unadjusted `matrix`.
    if apply_dual_candidate_filter:
        duals_vector = np.array([duals.get(j, 0.0) for j in range(n)])
        qaoa_matrix = matrix - duals_vector[np.newaxis, :]
    else:
        qaoa_matrix = matrix

    priced = []
    for curr_node in start_nodes:
        exclude = {depot_idx} if curr_node != depot_idx else set()
        k_batch = min(qubit_count, n - 1 - len(exclude))
        if k_batch <= 0:
            continue

        candidates = _nearest_and_explore_candidates(
            curr_node, exclude, matrix, k_batch, exploration_percent, rng,
            duals=candidate_duals, global_max_dist=global_max_dist,
        )
        if not candidates:
            continue

        try:
            subtour = solve_wslr_qaoa_subtour(curr_node, candidates, qaoa_matrix, xy_mixer=xy_mixer)
        except RuntimeError:
            raise  # e.g. Qiskit missing -- surface this clearly, don't swallow it
        if not subtour:
            continue

        full_nodes = [curr_node] + subtour
        for L in range(len(full_nodes), 0, -1):
            seg = full_nodes[:L]
            cost = _open_path_cost(seg, matrix)  # raw matrix -- real cost, not dual-adjusted
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
    n_iterations=ITERATION_CG,
):
    """
    Column-generation Open TSP solver: PuLP set-partitioning master problem
    + WS-LR QAOA sub-tour solver as the (heuristic) pricing subproblem.
    Runs `n_iterations` pricing iterations (LP relaxation for duals ->
    price -> grow pool), stopping early if a pass adds no new columns,
    then ONE final binary ILP over the accumulated pool. Returns the same
    shape of result dict as run_algo_hybrid_2_5, plus CG-specific
    diagnostics including a per-iteration log.
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
    n_iterations = max(1, n_iterations)

    node_ids = list(range(n))
    global_max_dist = float(matrix.max()) if n > 1 else 0.0

    # --- Pricing iterations: LP relaxation for duals -> price -> grow pool ---
    pool = _build_initial_columns(n, matrix, depot_idx)
    duals = {}
    iteration_log = []

    for it in range(1, n_iterations + 1):
        status_lp, _, _, duals = _solve_master(pool, node_ids, relaxation=True, time_limit=time_limit)
        if status_lp not in ("Optimal", "Not Solved"):
            warnings.warn(f"Iteration {it}: LP relaxation status was '{status_lp}' (expected 'Optimal').")

        priced_columns = _generate_priced_columns(
            matrix, depot_idx, duals, qubit_count, exploration_percent, xy_mixer,
            only_improving_columns, max_pricing_nodes, rng,
            apply_dual_candidate_filter=(it >= 2), global_max_dist=global_max_dist,
        )
        pool_before = len(pool)
        pool = _dedupe_columns(pool + priced_columns)
        n_new = len(pool) - pool_before

        iteration_log.append({
            "iteration": it,
            "lp_status": status_lp,
            "num_priced": len(priced_columns),
            "num_new_columns": n_new,
            "pool_size": len(pool),
        })

        if n_new == 0:
            break  # converged: pool unchanged, further iterations would be identical

    full_pool = pool

    # --- Final round: binary ILP over the accumulated pool ---
    status_final, selected_idx, _, _ = _solve_master(full_pool, node_ids, relaxation=False, time_limit=time_limit)

    if status_final == "Optimal" and selected_idx:
        selected_columns = [full_pool[i] for i in selected_idx]
    else:
        warnings.warn(
            f"Final ILP master status was '{status_final}'; falling back to a "
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
            "n_iterations": n_iterations,
        },
        "cg_diagnostics": {
            "num_iterations_run": len(iteration_log),
            "iteration_log": iteration_log,
            "final_master_status": status_final,
            "num_initial_columns": n,
            "num_pool_columns_final": len(full_pool),
            "num_segments_selected": len(selected_columns),
            "pre_2opt_cost": float(_open_path_cost(raw_tour, matrix)),
            "final_duals": duals,
        },
    }
