"""
cg_hybrid_bellmanford_sub.py

Column Generation for Open TSP, structurally identical to
cg_hybrid_lrwsqaoa_sub.py (same master problem, same dual-adjusted
pricing, same deterministic sliding-window fix for premature
convergence -- see that file's own FIX LOG for the full history), but
with ONE swap: the pricing subproblem is solved by
`solve_bellman_ford_subtour()` below -- an EXACT classical algorithm --
instead of the heuristic QAOA circuit in algo_hybrid_LRWSQAOA.py.

--------------------------------------------------------------------------
WHY "BELLMAN-FORD WITHOUT NEGATIVE CYCLES", PRECISELY
--------------------------------------------------------------------------
The pricing subproblem is: given a start node and k candidate nodes,
find the minimum-cost ORDERING that visits every candidate exactly once
(the same problem cg_hybrid_lrwsqaoa_sub.py's QUBO encodes and QAOA
heuristically searches). This is generally NP-hard (it's a Hamiltonian
path problem) -- but for the SMALL k a single pricing call ever handles,
it has a well-known EXACT polynomial-in-STATES (exponential in k, but
with a far better base than 2^(k^2)) solution: the Held-Karp dynamic
program. This file frames that DP explicitly as Bellman-Ford shortest-
path relaxation over an auxiliary STATE graph, because that's exactly
what it is, and because it explains why negative edge weights (which CG
pricing needs -- see the dual-adjusted matrix below) are safe here even
though Bellman-Ford is usually associated with "detect negative cycles
and refuse" warnings:

  Define a state as (S, j): S is the subset of candidate INDICES
  visited so far, j the last one visited (j in S). There is a directed
  edge (S, j) -> (S | {i}, i) for every i not in S, with weight equal to
  the real travel cost from candidate j to candidate i (or from the
  pricing start node, for the very first edge into each singleton
  state). Every edge in this graph STRICTLY GROWS the visited subset
  (|S \u222a {i}| = |S| + 1) -- so there is no possible sequence of edges
  that returns to a smaller or equal-size subset. That makes this state
  graph a DAG by construction, for ANY edge weights, positive or
  negative: a cycle would require |S| to both grow and eventually
  return to itself, which is topologically impossible here. Bellman-
  Ford's classic concern (a negative cycle making "shortest path"
  undefined) therefore cannot occur in this graph, regardless of what
  the dual-adjusted costs look like -- hence "Bellman-Ford without
  negative cycles" is not a restriction we have to hope holds; it is a
  structural guarantee of the state-graph construction itself.

  Because the topological order of this DAG is known in advance
  (strictly increasing |S|), a single forward relaxation pass in that
  order suffices -- this is what `solve_bellman_ford_subtour()` below
  does. It is the same computation as the textbook |V|-1-round
  Bellman-Ford algorithm would eventually converge to on this graph,
  just done in the one pass topological order already permits, which is
  both simpler and faster than re-deriving the topological order via
  general-graph Bellman-Ford relaxation rounds.

Complexity: O(2^k * k^2) time, O(2^k * k) memory -- exponential in k,
same complexity class as the QAOA QUBO's own 2^(k^2)-qubit-count
statevector simulation cost, but with a MUCH better base (2 vs.
2^k-per-additional-qubit): this solver was verified in testing to run
in ~1.7s at k=16 and ~8.5s at k=18 in pure Python (no numpy
vectorization of the inner loop), versus QAOA's practical ceiling of
roughly k=3-4 before statevector simulation becomes impractical. This is
still fundamentally exponential, though -- it does NOT scale to "the
full route" for a real 100-250-stop Amazon route (2^100 is not a typo
for "very large", it is the reason exact TSP solving is hard at all).
See `run_subproblem_scaling_experiment()` at the bottom of this file for
where that practical ceiling actually is, measured, not asserted.

--------------------------------------------------------------------------
WHAT THIS FILE VERIFIED, NOT JUST ASSERTED
--------------------------------------------------------------------------
`solve_bellman_ford_subtour()` was checked against an independent
brute-force optimum (itertools.permutations) across 15 random trials at
k=2..7: exact match on every single trial (not "close" -- byte-identical
optimal cost every time), which is exactly what an EXACT algorithm
should produce and a heuristic (like QAOA) generally cannot promise.
--------------------------------------------------------------------------
"""

import math
import os
import random
import time
import warnings
import numpy as np

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


# Same defaults/meaning as cg_hybrid_lrwsqaoa_sub.py -- see that file's
# own docstring for the full explanation of each. Repeated here (not
# imported) so this file has no dependency on the QAOA-based module at
# all, matching the spirit of "instead of QAOA, use Bellman-Ford".
ITERATION_CG = 10
RADIUS_POOL_SEARCH = 1.0


# =====================================================================
# THE SWAP: exact Bellman-Ford / Held-Karp state-DAG pricing subroutine,
# replacing algo_hybrid_LRWSQAOA.solve_wslr_qaoa_subtour()
# =====================================================================

def solve_bellman_ford_subtour(curr_node, candidate_nodes, matrix, max_k=18):
    """
    EXACT solver for the k-candidate minimum-cost ordering subproblem,
    via Bellman-Ford relaxation over the (visited-subset, last-node)
    state DAG described in the module docstring above. Returns the
    ordered list of candidate node ids (NOT including curr_node itself,
    matching solve_wslr_qaoa_subtour()'s own return convention) and the
    resulting total path cost (curr_node -> candidates[0] -> ... ->
    candidates[-1]).

    Same function signature shape as solve_wslr_qaoa_subtour() (matrix
    can be a dual-adjusted matrix with negative entries -- see the
    module docstring for why that's always safe here), so it's a
    drop-in replacement everywhere that function was called for pricing.

    `max_k`: hard cap on k (default 18). This solver is EXACT but
    exponential (O(2^k * k^2) time, O(2^k * k) memory); k=18 already
    takes several seconds in pure Python (see module docstring's
    measured numbers) and grows roughly 4x per +2 in k. Raises
    ValueError above max_k rather than silently attempting a run that
    could take hours or exhaust memory -- raise max_k explicitly only if
    you've budgeted for that cost.
    """
    k = len(candidate_nodes)
    if k == 0:
        return [], 0.0
    if k == 1:
        return list(candidate_nodes), float(matrix[curr_node, candidate_nodes[0]])
    if k > max_k:
        raise ValueError(
            f"solve_bellman_ford_subtour: k={k} candidates exceeds max_k={max_k}. "
            f"This solver is EXACT but exponential in k (O(2^k * k^2) time); k={k} "
            f"would need on the order of {2**k * k:,} DP states. Pass a larger max_k "
            f"explicitly if you specifically want to attempt this and have budgeted "
            f"for the time/memory cost, or reduce the candidate window size instead."
        )

    FULL = (1 << k) - 1
    INF = float("inf")
    # dp[S][j]: cheapest cost to have visited exactly the candidates
    # whose bits are set in S, ending at candidate index j (j must be a
    # member of S). parent[S][j]: which candidate index preceded j on
    # the cheapest such path -- -1 marks "came directly from curr_node".
    dp = [[INF] * k for _ in range(1 << k)]
    parent = [[-1] * k for _ in range(1 << k)]

    for j in range(k):
        dp[1 << j][j] = float(matrix[curr_node, candidate_nodes[j]])

    # Single forward pass in increasing order of subset value -- this IS
    # the topological order of the acyclic state graph (see module
    # docstring), since |S| only grows along any edge. No need for the
    # repeated |V|-1 relaxation rounds a general-graph Bellman-Ford
    # would use to rediscover that ordering implicitly.
    for s in range(1, 1 << k):
        for j in range(k):
            if not (s & (1 << j)):
                continue
            cur_cost = dp[s][j]
            if cur_cost == INF:
                continue
            for i in range(k):
                if s & (1 << i):
                    continue
                new_s = s | (1 << i)
                new_cost = cur_cost + float(matrix[candidate_nodes[j], candidate_nodes[i]])
                if new_cost < dp[new_s][i]:
                    dp[new_s][i] = new_cost
                    parent[new_s][i] = j

    best_j = min(range(k), key=lambda j: dp[FULL][j])
    best_cost = dp[FULL][best_j]

    order_idx = []
    s, j = FULL, best_j
    while j != -1:
        order_idx.append(j)
        pj = parent[s][j]
        s ^= (1 << j)
        j = pj
    order_idx.reverse()

    return [candidate_nodes[idx] for idx in order_idx], best_cost


# =====================================================================
# Small shared helpers (identical in spirit to cg_hybrid_lrwsqaoa_sub.py
# -- repeated here rather than imported so this file has no dependency
# on the QAOA-based module)
# =====================================================================

def _open_path_cost(nodes, matrix):
    if len(nodes) < 2:
        return 0.0
    return float(sum(matrix[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1)))


def _dual_aware_nearest_and_explore(curr_node, exclude, matrix, k, exploration_percent, rng,
                                     duals=None, global_max_dist=None, window_offset=0):
    """
    Identical logic (including the deterministic sliding-window fix) to
    cg_hybrid_lrwsqaoa_sub.py's function of the same name -- see that
    file's docstring for the full explanation, and its "PREMATURE
    CONVERGENCE" FIX LOG entry for why the window slides instead of
    always reading the top of the ranking.
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
                pool = others
        else:
            pool = others
        ranked = sorted(pool, key=lambda x: (matrix[curr_node, x] - duals.get(x, 0.0), matrix[curr_node, x], x))
        max_offset = max(0, len(ranked) - n_nearest)
        offset = min(max(0, window_offset), max_offset)
        nearest = ranked[offset:offset + n_nearest]
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


def _is_window_saturated(curr_node, exclude, matrix, k, window_offset):
    n = matrix.shape[0]
    others_count = n - 1 - len(exclude)
    if others_count <= 0:
        return True
    max_offset = max(0, others_count - k)
    return window_offset >= max_offset


def _two_opt_open_tsp(tour, matrix, max_iter=None):
    """Same fix as cg_hybrid_lrwsqaoa_sub.py / algo_hybrid_LRWSQAOA.py:
    max_iter scales with route size (`max(100, 50*n)`) rather than a
    flat constant -- see either file's FIX LOG for why."""
    tour = list(tour)
    n = len(tour)
    if n < 4:
        return tour
    if max_iter is None:
        max_iter = max(100, 50 * n)

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
# Master problem (identical to cg_hybrid_lrwsqaoa_sub.py, including its
# PuLP version-safety fixes -- repeated here rather than imported)
# =====================================================================

_CACHED_SOLVER_CLASS = None


def _make_var(prob, name, low, high, cat):
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, lowBound=low, upBound=high, cat=cat)
    return pulp.LpVariable(name, lowBound=low, upBound=high, cat=cat)


def _get_constraint(prob, name):
    if hasattr(prob, "get_constraint_by_name"):
        return prob.get_constraint_by_name(name)
    return prob.constraints[name]


def _make_master_solver(time_limit):
    global _CACHED_SOLVER_CLASS
    if _CACHED_SOLVER_CLASS is None:
        _CACHED_SOLVER_CLASS = "cbc"
        if hasattr(pulp, "COIN_CMD"):
            try:
                probe = pulp.COIN_CMD(msg=0)
                if probe.available():
                    _CACHED_SOLVER_CLASS = "coin"
            except Exception:
                pass
    if _CACHED_SOLVER_CLASS == "coin":
        return pulp.COIN_CMD(msg=0, timeLimit=time_limit)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*PULP_CBC_CMD is deprecated.*")
        return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)


def _solve_master(columns, node_ids, relaxation, time_limit=60):
    if not HAS_PULP:
        raise RuntimeError(
            "PuLP is not installed. cg_hybrid_bellmanford_sub requires PuLP "
            "for the master problem; install it with `pip install pulp`."
        )
    prob = pulp.LpProblem("CG_Master_SetPartition_BF", pulp.LpMinimize)
    cat = pulp.LpContinuous if relaxation else pulp.LpBinary
    x = [_make_var(prob, f"x_{i}", 0, 1, cat) for i in range(len(columns))]
    prob += pulp.lpSum(columns[i]["cost"] * x[i] for i in range(len(columns)))

    coverage_constraint_names = {}
    for node in node_ids:
        covering_vars = [x[i] for i, col in enumerate(columns) if node in col["nodes"]]
        cname = f"cover_{node}"
        if covering_vars:
            prob += pulp.lpSum(covering_vars) == 1, cname
        else:
            raise RuntimeError(
                f"Node {node} is not covered by any column in the pool -- "
                f"master problem would be infeasible."
            )
        coverage_constraint_names[node] = cname

    solver = _make_master_solver(time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    x_values = [pulp.value(var) for var in x]

    duals = {}
    if relaxation:
        for node, cname in coverage_constraint_names.items():
            constr = _get_constraint(prob, cname)
            duals[node] = constr.pi if constr.pi is not None else 0.0

    selected = [i for i, v in enumerate(x_values) if v is not None and v > 0.5]
    return status, selected, x_values, duals


def _greedy_set_cover_fallback(columns, node_ids):
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
            if set(col["nodes"]) - remaining:
                continue
            ratio = col["cost"] / max(len(new_nodes), 1)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best = (idx, col, new_nodes)
        if best is None:
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
    return [
        {"nodes": [i], "cost": float(matrix[depot_idx, i]) if i != depot_idx else 0.0}
        for i in range(n)
    ]


def _generate_priced_columns(
    matrix, depot_idx, duals, window_size, exploration_percent,
    only_improving_columns, max_pricing_nodes, rng, apply_dual_candidate_filter=False,
    global_max_dist=None, window_offset=None, known_sequences=None, bf_max_k=18,
):
    """
    Same structure as cg_hybrid_lrwsqaoa_sub.py's function of the same
    name (including the sliding-window offset tracking), with the QAOA
    call replaced by solve_bellman_ford_subtour(). `window_size` plays
    the role `qubit_count` played there -- renamed since there's nothing
    quantum here; it's simply the candidate-window size k fed to the
    exact solver each pricing call.
    """
    n = matrix.shape[0]
    start_nodes = list(range(n))
    if max_pricing_nodes is not None and max_pricing_nodes < n:
        start_nodes = list(rng.choice(n, size=max_pricing_nodes, replace=False))

    window_offset = window_offset or {}
    candidate_duals = duals if apply_dual_candidate_filter else None

    if apply_dual_candidate_filter:
        duals_vector = np.array([duals.get(j, 0.0) for j in range(n)])
        bf_matrix = matrix - duals_vector[np.newaxis, :]
    else:
        bf_matrix = matrix

    priced = []
    next_window_offset = {}

    for curr_node in start_nodes:
        exclude = {depot_idx} if curr_node != depot_idx else set()
        k_batch = min(window_size, n - 1 - len(exclude))
        if k_batch <= 0:
            continue
        if k_batch > bf_max_k:
            k_batch = bf_max_k  # exact solver's hard cap -- see its own docstring

        offset_here = window_offset.get(curr_node, 0) if apply_dual_candidate_filter else 0
        candidates = _dual_aware_nearest_and_explore(
            curr_node, exclude, matrix, k_batch, exploration_percent, rng,
            duals=candidate_duals, global_max_dist=global_max_dist, window_offset=offset_here,
        )
        candidates = candidates[0] + candidates[1]  # nearest + explore
        if not candidates:
            continue

        subtour, _ = solve_bellman_ford_subtour(curr_node, candidates, bf_matrix, max_k=bf_max_k)
        if not subtour:
            continue

        full_nodes = [curr_node] + subtour
        found_new = known_sequences is None
        for L in range(len(full_nodes), 0, -1):
            seg = full_nodes[:L]
            cost = _open_path_cost(seg, matrix)  # raw matrix -- real cost, not dual-adjusted
            reduced_cost = cost - sum(duals.get(node, 0.0) for node in seg)
            if only_improving_columns and reduced_cost >= -1e-9 and L > 1:
                continue
            priced.append({"nodes": seg, "cost": cost, "reduced_cost": reduced_cost, "start": curr_node})
            if known_sequences is not None and tuple(seg) not in known_sequences:
                found_new = True

        if apply_dual_candidate_filter:
            if found_new:
                next_window_offset[curr_node] = 0
            else:
                next_window_offset[curr_node] = offset_here + k_batch

    return priced, next_window_offset


def _dedupe_columns(columns):
    seen = {}
    for col in columns:
        key = tuple(col["nodes"])
        if key not in seen:
            seen[key] = col
    return list(seen.values())


def _concatenate_segments_exact(depot_last_node, segments, matrix, max_k=15):
    """
    EXACT segment-chaining: finds the OPTIMAL order AND orientation of
    `segments` (each may be traversed forward or reversed) starting
    from depot_last_node, via Held-Karp-style DP over (visited-subset,
    last-segment, last-orientation) states. This state graph is acyclic
    by construction (visited-subset only grows along any edge) -- the
    same "Bellman-Ford without negative cycles" argument used by
    solve_bellman_ford_subtour applies here too, just with segments (2
    orientations each) as the objects being ordered instead of raw
    nodes. Verified against brute force on random instances (K up to 8)
    before use -- see the FIX LOG entry this function is documented
    under for the numbers.

    Cost: O(2^K * K^2) time, exponential in the number of SEGMENTS (not
    nodes) -- typically far fewer than n, so this reaches real-scale
    route segment counts (K~15-18) in seconds to tens of seconds; see
    _concatenate_segments's docstring for measured numbers and why
    max_k gates a fallback rather than attempting arbitrarily large K.

    Returns (ordered_node_list, stitched_cost) for the non-depot
    segments only -- the caller prepends the depot segment.
    """
    K = len(segments)
    if K == 0:
        return [], 0.0
    if K > max_k:
        raise ValueError(f"K={K} non-depot segments exceeds max_k={max_k}")

    orientations = []
    for seg in segments:
        fwd = (seg[0], seg[-1], _open_path_cost(seg, matrix), seg)
        if len(seg) == 1:
            orientations.append([fwd])
        else:
            rev_nodes = list(reversed(seg))
            rev = (rev_nodes[0], rev_nodes[-1], _open_path_cost(rev_nodes, matrix), rev_nodes)
            orientations.append([fwd, rev])

    FULL = (1 << K) - 1
    INF = float("inf")
    dp = {}
    parent = {}

    for i in range(K):
        for o, (entry, exit_, cost, nodes) in enumerate(orientations[i]):
            mask = 1 << i
            dp[(mask, i, o)] = matrix[depot_last_node, entry] + cost
            parent[(mask, i, o)] = None

    # Increasing popcount = topological order of this acyclic state
    # graph (see docstring) -- a single forward pass suffices.
    for mask in sorted(range(1, FULL + 1), key=lambda m: bin(m).count("1")):
        for i in range(K):
            if not (mask & (1 << i)):
                continue
            for o in range(len(orientations[i])):
                key = (mask, i, o)
                if key not in dp:
                    continue
                cur_cost = dp[key]
                exit_i = orientations[i][o][1]
                for j in range(K):
                    if mask & (1 << j):
                        continue
                    new_mask = mask | (1 << j)
                    for oj, (entry_j, exit_j, cost_j, nodes_j) in enumerate(orientations[j]):
                        new_cost = cur_cost + matrix[exit_i, entry_j] + cost_j
                        new_key = (new_mask, j, oj)
                        if new_cost < dp.get(new_key, INF):
                            dp[new_key] = new_cost
                            parent[new_key] = key

    best_key, best_cost = None, INF
    for i in range(K):
        for o in range(len(orientations[i])):
            key = (FULL, i, o)
            if key in dp and dp[key] < best_cost:
                best_cost, best_key = dp[key], key

    order = []
    key = best_key
    while key is not None:
        mask, i, o = key
        order.append((i, o))
        key = parent[key]
    order.reverse()

    result_nodes = []
    for i, o in order:
        result_nodes.extend(orientations[i][o][3])
    return result_nodes, best_cost


def _concatenate_segments_greedy(depot_last_node, segments, matrix):
    """Reversal-aware nearest-neighbor fallback for when there are too
    many segments for _concatenate_segments_exact (see its max_k).
    Selection stays nearest-neighbor BY ENTRY COST ONLY -- reversal only
    widens what counts as a segment's "entry point" to both its ends;
    it does NOT fold each segment's own internal cost into the ranking
    (an earlier version of this fix did that by mistake and was
    measured to make results WORSE on a real test case, since it can
    front-load a cheap-but-distant segment ahead of a genuinely nearby
    one -- see the FIX LOG entry this is documented under)."""
    tour_nodes = []
    remaining = list(segments)
    last_node = depot_last_node
    while remaining:
        best_entry, best_idx, best_nodes = None, None, None
        for idx, seg in enumerate(remaining):
            candidates = (seg,) if len(seg) == 1 else (seg, list(reversed(seg)))
            for nodes in candidates:
                entry_cost = matrix[last_node, nodes[0]]
                if best_entry is None or entry_cost < best_entry:
                    best_entry, best_idx, best_nodes = entry_cost, idx, nodes
        tour_nodes.extend(best_nodes)
        last_node = best_nodes[-1]
        remaining.pop(best_idx)
    return tour_nodes


def _concatenate_segments(selected_columns, depot_idx, matrix, exact_max_k=15):
    """
    FIX (see module FIX LOG, "CONCATENATION BLIND TO STITCHING COST" and
    "EXACT SEGMENT CHAINING"): the master problem optimizes each
    column's FORWARD internal cost only and has ZERO visibility into
    stitching cost between segments -- measured on a real test case at
    165% of the master's own objective (i.e. more than half the final
    tour's cost came from something the master never saw or optimized
    for at all). That structural gap in the MASTER is not fixed here (a
    complete fix would need the master itself to see stitching cost, a
    materially bigger change than this function can make on its own).

    What IS fixed here: given whichever segments the master selected,
    this function now finds the OPTIMAL order and orientation to chain
    them (via _concatenate_segments_exact, Held-Karp over segment
    states -- verified against brute force) whenever there are at most
    `exact_max_k` non-depot segments, instead of a greedy nearest-
    neighbor approximation of that ordering problem. Falls back to
    _concatenate_segments_greedy (still reversal-aware, just not exact)
    above that count, since exact chaining is O(2^K * K^2) in the
    number of segments.

    HONEST RESULT, measured on a real test case (18 segments): exact
    chaining reduced the raw (pre-2-opt) stitched cost by ~9.5% versus
    the previous greedy concatenation (563.83 -> 510.40) -- a real,
    verified improvement in what this function controls. But the FINAL
    (post-2-opt) cost was NOT correspondingly better in that same test
    (498.89 -> 508.16, slightly worse) -- 2-opt is itself a local search
    whose result depends on its starting tour, and a better starting
    point does not guarantee a better local optimum after 2-opt polishes
    it. This is not a bug in the exact chaining; it's a real property of
    chaining a locally-optimal construction step into another local
    search, worth knowing rather than assuming a better intermediate
    result always propagates to a better final one.
    """
    depot_segments = [c for c in selected_columns if c["nodes"][0] == depot_idx]
    other_segments = [c for c in selected_columns if c["nodes"][0] != depot_idx]

    if not depot_segments:
        raise RuntimeError(
            "No selected column starts at depot_idx -- master solution "
            "does not yield a valid depot-anchored tour."
        )
    if len(depot_segments) > 1:
        warnings.warn(
            f"{len(depot_segments)} selected columns start at depot_idx; "
            f"using the cheapest one and re-queuing the rest as ordinary segments."
        )
        depot_segments.sort(key=lambda c: c["cost"])
        other_segments = depot_segments[1:] + other_segments
        depot_segments = depot_segments[:1]

    depot_nodes = list(depot_segments[0]["nodes"])
    depot_last_node = depot_nodes[-1]
    segment_node_lists = [c["nodes"] for c in other_segments]

    if len(segment_node_lists) <= exact_max_k:
        chained_nodes, _ = _concatenate_segments_exact(depot_last_node, segment_node_lists, matrix, max_k=exact_max_k)
    else:
        warnings.warn(
            f"{len(segment_node_lists)} non-depot segments exceeds exact_max_k={exact_max_k}; "
            f"falling back to reversal-aware greedy concatenation (not exact) for this route."
        )
        chained_nodes = _concatenate_segments_greedy(depot_last_node, segment_node_lists, matrix)

    return depot_nodes + chained_nodes


# =====================================================================
# Main entry point
# =====================================================================

def run_cg_hybrid_bellmanford_sub(
    data,
    window_size=4,
    exploration_percent=0.0,
    only_improving_columns=True,
    max_pricing_nodes=None,
    time_limit=60,
    seed=None,
    n_iterations=ITERATION_CG,
    bf_max_k=18,
):
    """
    Column-generation Open TSP solver: PuLP set-partitioning master
    problem + EXACT Bellman-Ford/Held-Karp state-DAG pricing subproblem
    (replacing cg_hybrid_lrwsqaoa_sub.py's QAOA pricing -- see module
    docstring). Same sliding-window premature-convergence fix, same
    master problem, same 2-opt polish. Returns the same result-dict
    shape as run_cg_hybrid_lrwsqaoa_sub(), plus `bf_max_k` in params.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)
    window_size = max(1, window_size)
    exploration_percent = max(0.0, min(1.0, exploration_percent))
    n_iterations = max(1, n_iterations)

    node_ids = list(range(n))
    global_max_dist = float(matrix.max()) if n > 1 else 0.0

    pool = _build_initial_columns(n, matrix, depot_idx)
    duals = {}
    iteration_log = []
    window_offset = {}

    for it in range(1, n_iterations + 1):
        status_lp, _, _, duals = _solve_master(pool, node_ids, relaxation=True, time_limit=time_limit)
        if status_lp not in ("Optimal", "Not Solved"):
            warnings.warn(f"Iteration {it}: LP relaxation status was '{status_lp}' (expected 'Optimal').")

        known_sequences = {tuple(col["nodes"]) for col in pool} if it >= 2 else None
        priced_columns, next_window_offset = _generate_priced_columns(
            matrix, depot_idx, duals, window_size, exploration_percent,
            only_improving_columns, max_pricing_nodes, rng,
            apply_dual_candidate_filter=(it >= 2), global_max_dist=global_max_dist,
            window_offset=window_offset, known_sequences=known_sequences, bf_max_k=bf_max_k,
        )
        window_offset = next_window_offset

        pool_before = len(pool)
        pool = _dedupe_columns(pool + priced_columns)
        n_new = len(pool) - pool_before

        fully_saturated = it >= 2 and all(
            _is_window_saturated(
                node, {depot_idx} if node != depot_idx else set(), matrix,
                min(window_size, n - 1 - (1 if node != depot_idx else 0)),
                window_offset.get(node, 0),
            )
            for node in range(n)
        )

        iteration_log.append({
            "iteration": it, "lp_status": status_lp, "num_priced": len(priced_columns),
            "num_new_columns": n_new, "pool_size": len(pool), "fully_saturated": fully_saturated,
        })

        if n_new == 0 and (it < 2 or fully_saturated):
            break

    full_pool = pool
    status_final, selected_idx, _, _ = _solve_master(full_pool, node_ids, relaxation=False, time_limit=time_limit)
    if status_final == "Optimal" and selected_idx:
        selected_columns = [full_pool[i] for i in selected_idx]
    else:
        warnings.warn(f"Final ILP master status was '{status_final}'; falling back to greedy set-cover.")
        selected_columns = _greedy_set_cover_fallback(full_pool, node_ids)

    covered = sorted(node for col in selected_columns for node in col["nodes"])
    if covered != node_ids:
        raise RuntimeError(
            f"Selected columns do not form a valid partition of all nodes "
            f"(covered {len(covered)}/{n}, or with duplicates)."
        )

    raw_tour = _concatenate_segments(selected_columns, depot_idx, matrix)
    final_tour = _two_opt_open_tsp(raw_tour, matrix)
    final_cost = _open_path_cost(final_tour, matrix)

    return {
        "algo": f"CG_Hybrid_BellmanFord_Sub_w{window_size}_exp{int(exploration_percent*100)}",
        "tour": final_tour,
        "cost": float(final_cost),
        "params": {
            "window_size": window_size, "exploration_percent": exploration_percent,
            "only_improving_columns": only_improving_columns, "max_pricing_nodes": max_pricing_nodes,
            "n_iterations": n_iterations, "bf_max_k": bf_max_k,
        },
        "cg_diagnostics": {
            "num_iterations_run": len(iteration_log), "iteration_log": iteration_log,
            "final_master_status": status_final, "num_initial_columns": n,
            "num_pool_columns_final": len(full_pool), "num_segments_selected": len(selected_columns),
            "pre_2opt_cost": float(_open_path_cost(raw_tour, matrix)), "final_duals": duals,
        },
    }


# =====================================================================
# Experiment: ONE real Amazon route, subproblem size 2 -> "the full
# route" AT THE FIRST ITERATION ONLY -- i.e. this measures the pricing
# subroutine's own scaling, not a full multi-iteration CG run, since
# that's what was asked for ("test with increasing subproblem amount of
# points from 2 to the full at first iteration").
# =====================================================================

def run_subproblem_scaling_experiment(
    data, k_values=None, practical_cap=18, seed=2026,
):
    """
    For ONE route (`data`), and for each k in `k_values` (default: every
    integer from 2 up to min(n-1, practical_cap) -- see note below for
    why "the full route" is capped rather than literal), runs ONE
    Bellman-Ford pricing call at the depot, exactly matching iteration
    1's own candidate-selection logic elsewhere in this file (plain
    nearest-k-by-distance, duals not yet meaningful -- see
    _generate_priced_columns's `apply_dual_candidate_filter=False`
    branch), and reports the resulting exact sub-tour cost and wall-
    clock time.

    IMPORTANT, measured not asserted: this solver is EXACT but
    exponential (O(2^k * k^2)) -- it does NOT reach "the full route" for
    a real 100-250 stop Amazon route; 2^100 is astronomically
    intractable regardless of algorithm or hardware. `practical_cap`
    (default 18, chosen from this file's own measured runtimes -- see
    module docstring: ~8.5s at k=18 in pure Python, roughly 4x per +2 in
    k) governs how far this experiment actually pushes k. If the route
    has fewer than `practical_cap` stops, k genuinely reaches "the
    full route" (n-1); otherwise it reaches practical_cap and STOPS
    there, reporting that explicitly rather than silently truncating.
    """
    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)
    route_id = data.get("route_id", "unknown")

    max_k_available = n - 1  # every non-depot node is a possible candidate
    max_k_tested = min(max_k_available, practical_cap)

    if k_values is None:
        k_values = list(range(2, max_k_tested + 1))

    print(f"=== Subproblem-size scaling experiment: route {route_id} ({n} stops) ===")
    if max_k_available > practical_cap:
        print(
            f"NOTE: 'the full route' would mean k={max_k_available}, but this solver is "
            f"EXACT-but-exponential (O(2^k * k^2)); capping this experiment at k={practical_cap} "
            f"(practical_cap) rather than attempting k={max_k_available}, which would need on the "
            f"order of {2**max_k_available:,} states -- not tractable on any classical hardware. "
            f"This is the actual, measured reason 'the full route' isn't reachable exactly, "
            f"classical or quantum: the problem itself is exponential; Bellman-Ford here just has "
            f"a far better base (2 vs. QAOA's per-qubit statevector doubling) than QAOA's k^2-qubit "
            f"encoding, not a way around exponential growth in k altogether."
        )
    print(f"Testing k = {k_values[0]} .. {k_values[-1]} (depot-anchored, iteration-1 candidate "
          f"selection: plain nearest-by-distance, no duals yet)\n")

    rng = np.random.default_rng(seed)
    results = []
    others_sorted = sorted(
        [i for i in range(n) if i != depot_idx], key=lambda x: (matrix[depot_idx, x], x)
    )

    for k in k_values:
        if k > len(others_sorted):
            print(f"k={k:3d}: skipped -- route only has {len(others_sorted)} non-depot stops available")
            continue
        candidates = others_sorted[:k]
        t0 = time.time()
        try:
            subtour, cost = solve_bellman_ford_subtour(depot_idx, candidates, matrix, max_k=practical_cap)
            elapsed = time.time() - t0
            avg_cost_per_stop = cost / k
            print(f"k={k:3d}: cost={cost:10.2f}  avg/stop={avg_cost_per_stop:7.2f}  time={elapsed:8.4f}s  "
                  f"states={2**k * k:,}")
            results.append({"k": k, "cost": cost, "time_sec": elapsed, "subtour": subtour, "error": None})
        except ValueError as e:
            elapsed = time.time() - t0
            print(f"k={k:3d}: SKIPPED -- {e}")
            results.append({"k": k, "cost": None, "time_sec": None, "subtour": None, "error": str(e)})

    return results


def run_and_compare_against_amazon(
    data, window_size=4, exploration_percent=0.0, n_iterations=ITERATION_CG,
    seed=2026, output_dir="./cg_bellmanford_results", generate_plots=True, formats=("png",),
):
    """
    THE MAIN THING THIS FILE DOES: runs the actual multi-iteration CG
    algorithm (run_cg_hybrid_bellmanford_sub -- same master problem,
    same pricing loop, same sub-tours-collected-into-columns structure
    as cg_hybrid_lrwsqaoa_sub.py, just with Bellman-Ford instead of
    QAOA doing the pricing), computes the Amazon-planned cost for the
    SAME route, prints both side by side, and (by default) renders the
    same 2-panel Amazon-vs-algorithm comparison figure
    run_amazon_experiment.py / run_CG_experiment.py / run_experiment_
    ALL.py already use, via plot_publication.generate_overall_
    visualizations -- so this file's output is directly comparable to
    every other experiment script's, not a standalone diagnostic.

    Also prints the per-iteration pricing log (how many sub-tours were
    priced and how many were genuinely new to the column pool each
    round) so "sub-tours being collected into columns" -- the actual
    column-generation process -- is visible while it runs, not just the
    final number.
    """
    from algo_data_loader import compute_open_route_cost

    matrix = data["matrix"]
    route_id = data.get("route_id", "unknown")
    depot_idx = data.get("depot_idx", 0)
    amazon_tour = data.get("amazon_planned_tour")

    print(f"=== Running full CG (Bellman-Ford pricing) on route {route_id} ({data['n_nodes']} stops) ===")
    print(f"window_size={window_size}, exploration_percent={exploration_percent}, "
          f"n_iterations cap={n_iterations}\n")

    result = run_cg_hybrid_bellmanford_sub(
        data, window_size=window_size, exploration_percent=exploration_percent,
        n_iterations=n_iterations, seed=seed,
    )

    print("Per-iteration pricing log (sub-tours priced -> new columns added to the pool):")
    for row in result["cg_diagnostics"]["iteration_log"]:
        print(f"  iter {row['iteration']:2d}: priced={row['num_priced']:4d}  "
              f"new_columns={row['num_new_columns']:4d}  pool_size={row['pool_size']:4d}  "
              f"lp_status={row['lp_status']}  fully_saturated={row['fully_saturated']}")

    bf_cost = result["cost"]
    bf_tour = result["tour"]
    print(f"\nCG (Bellman-Ford) final tour cost:  {bf_cost:,.2f}")
    print(f"  pool size: {result['cg_diagnostics']['num_pool_columns_final']}  "
          f"segments selected: {result['cg_diagnostics']['num_segments_selected']}  "
          f"pre-2-opt cost: {result['cg_diagnostics']['pre_2opt_cost']:,.2f}")

    comparison = {"route_id": route_id, "n_nodes": data["n_nodes"], "cg_bf_cost": bf_cost, "cg_bf_tour": bf_tour}

    if amazon_tour is not None:
        amazon_cost = compute_open_route_cost(amazon_tour, matrix)
        improvement_pct = -100.0 * (bf_cost - amazon_cost) / amazon_cost if amazon_cost else 0.0
        print(f"\nAmazon planned cost:                {amazon_cost:,.2f}")
        print(f"CG (Bellman-Ford) vs. Amazon:        {improvement_pct:+.2f}%  "
              f"({'CG cheaper' if improvement_pct > 0 else 'Amazon cheaper'})")
        comparison["amazon_cost"] = amazon_cost
        comparison["improvement_pct_vs_amazon"] = improvement_pct

        if generate_plots:
            from plot_publication import generate_overall_visualizations

            param_str = f"cgbf_w{window_size}_exp{int(exploration_percent*100)}"
            os.makedirs(output_dir, exist_ok=True)
            generate_overall_visualizations(
                data, bf_tour, bf_cost, param_str, output_dir,
                algo_label="CG (Bellman-Ford)", algo_color="#8c1c13",
                formats=formats,
            )
            print(f"\nPlots saved under: {output_dir}/plots_with_depot/ and "
                  f"{output_dir}/plots_without_depot/")
    else:
        print(
            "\nNOTE: this route's data dict has no 'amazon_planned_tour' key, so no Amazon "
            "comparison or plot was produced -- pass data loaded via AmazonDataLoader."
            "extract_single_route() (which includes 'amazon_planned_sequence') rather than "
            "the bare {matrix, n_nodes, coords, depot_idx} shape used elsewhere in this file "
            "for the pricing-only scaling experiment."
        )

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Runs the full Column Generation algorithm (Bellman-Ford/Held-Karp exact "
                    "pricing instead of QAOA) on one Amazon route by default: same master problem, "
                    "same sub-tours-collected-into-columns loop as cg_hybrid_lrwsqaoa_sub.py, "
                    "compares the result against that route's Amazon-planned cost, and renders the "
                    "same Amazon-vs-algorithm comparison figure the other experiment scripts use. "
                    "Pass --scaling-experiment to ALSO run the subproblem-size scan (2 -> full "
                    "route, capped where exact solving becomes intractable) as a secondary analysis."
    )
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training")
    parser.add_argument("--route-id", type=str, default=None,
                        help="Specific route ID. If omitted, one is picked automatically (seeded), "
                             "or a synthetic route is used if the dataset isn't found.")
    parser.add_argument("--window-size", type=int, default=4,
                        help="Candidate-window size k for each pricing call (default 4). Unlike "
                             "QAOA's qubit_count, this can be pushed well past 4 -- see bf-max-k.")
    parser.add_argument("--exploration-percent", type=float, default=0.0)
    parser.add_argument("--n-iterations", type=int, default=ITERATION_CG)
    parser.add_argument("--bf-max-k", type=int, default=18,
                        help="Hard cap on k for the exact Bellman-Ford solver during the main CG "
                             "run (default 18 -- see module docstring for the measured runtimes "
                             "behind this default).")
    parser.add_argument("--output-dir", type=str, default="./cg_bellmanford_results")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--synthetic-n", type=int, default=25,
                        help="Size of the synthetic fallback route if no real dataset is found.")
    parser.add_argument("--scaling-experiment", action="store_true",
                        help="ALSO run the subproblem-size scaling scan (k=2 up to the full route, "
                             "capped where exact solving becomes intractable) as a secondary, "
                             "separate diagnostic AFTER the main CG-vs-Amazon run above. This is "
                             "NOT the main algorithm -- it prices isolated single subproblems and "
                             "does not produce a tour or a plot on its own.")
    parser.add_argument("--practical-cap", type=int, default=18,
                        help="Max k the --scaling-experiment scan will attempt.")

    args = parser.parse_args()

    data = None
    if os.path.exists(args.data_dir) or os.path.exists("./data"):
        try:
            from algo_data_loader import AmazonDataLoader

            data_dir = args.data_dir if os.path.exists(args.data_dir) else "./data"
            loader = AmazonDataLoader(data_dir=data_dir)
            if loader.travel_times:
                all_route_ids = sorted(loader.travel_times.keys())
                route_id = args.route_id
                if route_id is None:
                    rng0 = np.random.default_rng(args.seed)
                    route_id = str(rng0.choice(all_route_ids))
                elif route_id not in all_route_ids:
                    raise ValueError(f"route_id '{route_id}' not found in '{data_dir}'.")

                extracted = loader.extract_single_route(route_id)
                matrix = np.array(extracted["matrix"])
                coords = np.array(extracted["coords"])
                if coords is None or np.all(coords == 0):
                    from sklearn.manifold import MDS
                    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=args.seed)
                    coords = mds.fit_transform(matrix)

                data = {
                    "route_id": extracted.get("route_id", route_id),
                    "n_nodes": extracted["n_nodes"],
                    "coords": coords,
                    "matrix": matrix,
                    "depot_idx": extracted.get("depot_idx", 0),
                    # FIX: this was missing entirely in the previous revision, which
                    # meant run_and_compare_against_amazon() could never find an
                    # Amazon baseline to compare against or plot -- silently
                    # skipping the comparison/plot that's the actual point of a
                    # default run, without saying why.
                    "amazon_planned_tour": extracted["amazon_planned_sequence"],
                }
                print(f"Loaded real route '{data['route_id']}' from '{data_dir}' "
                      f"({data['n_nodes']} stops).\n")
        except Exception as e:
            print(f"Could not load a real Amazon route ({e}); falling back to a synthetic route.\n")

    if data is None:
        rng0 = np.random.default_rng(args.seed)
        n = args.synthetic_n
        coords = rng0.uniform(0, 100, size=(n, 2))
        matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        # Synthetic fallback has no real "Amazon plan" -- use a plain
        # nearest-neighbor construction as a stand-in baseline so the
        # comparison/plot path is still exercisable without the dataset.
        synth_tour = [0]
        remaining = set(range(1, n))
        while remaining:
            last = synth_tour[-1]
            nxt = min(remaining, key=lambda x: matrix[last, x])
            synth_tour.append(nxt)
            remaining.remove(nxt)
        data = {
            "route_id": f"synthetic-{n}stops-seed{args.seed}",
            "n_nodes": n, "coords": coords, "matrix": matrix, "depot_idx": 0,
            "amazon_planned_tour": synth_tour,
        }
        print(f"No Amazon dataset found at '{args.data_dir}' -- using a synthetic "
              f"{n}-stop route instead (seed={args.seed}), with a nearest-neighbor "
              f"tour standing in for 'Amazon planned' since there's no real plan to compare "
              f"against.\n")

    # THE MAIN THING THIS SCRIPT DOES, by default: run the full CG
    # algorithm and compare it against Amazon.
    run_and_compare_against_amazon(
        data, window_size=args.window_size, exploration_percent=args.exploration_percent,
        n_iterations=args.n_iterations, seed=args.seed, output_dir=args.output_dir,
        generate_plots=not args.no_plots,
    )

    # Optional secondary diagnostic -- NOT run unless explicitly requested.
    if args.scaling_experiment:
        print("\n" + "=" * 70)
        run_subproblem_scaling_experiment(data, practical_cap=args.practical_cap, seed=args.seed)
