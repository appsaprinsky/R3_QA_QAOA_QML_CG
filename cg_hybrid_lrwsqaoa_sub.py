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
    - The ranking WINDOW itself slides deterministically per node across
      iterations rather than always being the top of the list -- see
      "PREMATURE CONVERGENCE" in the FIX LOG below for why, and
      _dual_aware_nearest_and_explore's docstring for the mechanics.
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

  Iterations stop when EITHER `n_iterations` is reached, OR a pass adds
  zero new columns AND every node's ranking window has been slid all the
  way through its full candidate list (see "PREMATURE CONVERGENCE" in
  the FIX LOG below) -- not simply "this iteration's narrow window found
  nothing", which is a materially weaker and, on real instances,
  frequently premature condition.

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
FIX LOG (this revision)
--------------------------------------------------------------------------
* _solve_master(): TWO separate PuLP deprecation warnings were still
  firing (a previous revision already fixed a THIRD one, the
  LpVariable-construction warning, via _make_var below):
    1. `prob.constraints[cname]` (dict-style access) -- deprecated in
       favor of `prob.get_constraint_by_name(name)`. This was called
       ONCE PER NODE, every LP relaxation solve (i.e. up to n * 10 times
       per single run_cg_hybrid_lrwsqaoa_sub() call on a 250-stop
       route) -- at default warnings settings Python only shows/records
       a given warning once per call site and normally fast-paths
       identical repeats, but the "show every warning" wrapper used by
       run_experiment_ALL.py's warning capture (warnings.simplefilter
       ("always")) disabled that fast path, so those thousands of
       nearly-identical warnings were being individually constructed,
       formatted, and processed -- a real, measurable contributor to
       "the code got slower". Fixed via _get_constraint() below, which
       uses the new API when available and only falls back to the old
       dict access (with the warning intact) on PuLP versions that
       don't have get_constraint_by_name yet.
    2. `pulp.PULP_CBC_CMD` -- deprecated in favor of `pulp.COIN_CMD`
       (which needs the `pulp[cbc]` extra installed to find a CBC
       binary, so blindly switching to it could break environments that
       don't have that extra). Fixed via _make_master_solver() below:
       tries COIN_CMD once, caches whether it's actually usable at the
       MODULE level (not per solve -- probing solver availability on
       every single LP/ILP solve would itself be a performance
       regression, and _solve_master is called up to n_iterations+1
       times per run), and falls back to PULP_CBC_CMD (with its own
       deprecation warning locally suppressed, since the fallback here
       is intentional, not an oversight) if COIN_CMD isn't usable.

PREMATURE CONVERGENCE (this revision -- the actual issue reported: "CG
stops after not so many iterations, results are worse than Amazon, and
it's not about exploration_percent")
--------------------------------------------------------------------------
* Root cause: the "n_new == 0 -> converged" stopping rule
  (run_cg_hybrid_lrwsqaoa_sub's main loop) is only a valid convergence
  PROOF if pricing is an exact oracle that can certify no improving
  column exists anywhere. It isn't one here -- for every node,
  _dual_aware_nearest_and_explore only ever considered the literal TOP
  of the reduced-cost ranking (`ranked[:n_nearest]`, unconditionally).
  With exploration_percent at 0 (or low), that "nearest" slice is fully
  DETERMINISTIC given the current duals. Once the LP's duals stabilize
  between iterations -- which happens naturally and fairly quickly, once
  new columns stop changing the LP's optimal basis, a self-reinforcing
  fixed point -- pricing re-proposes the EXACT SAME top-k window every
  time, re-derives the exact same truncations, finds them all already
  pooled, `n_new` hits 0, and the run reports "converged" -- while having
  never once looked at the (k+1)-th, 10th, or 50th-ranked candidate for
  any node. This is "the heuristic's narrow window hit a fixed point",
  not "no improving column exists" -- a materially different, much
  weaker claim that the old code was silently treating as equivalent.
  (See the empirical verification note at the end of this entry.)
* FIX: candidate selection now uses a per-node ranking window that
  SLIDES deterministically (no randomness -- unrelated to and untouched
  from exploration_percent, per explicit instruction) instead of always
  reading the top of the list:
    - `_dual_aware_nearest_and_explore` gained a `window_offset`
      parameter: `nearest = ranked[window_offset : window_offset + k]`
      instead of always `ranked[:k]`.
    - `run_cg_hybrid_lrwsqaoa_sub` now maintains a persistent
      `{node: offset}` dict across iterations. After each pricing round,
      `_generate_priced_columns` reports, per node, whether ITS
      truncations included anything genuinely new (not already in the
      pool before this round) -- a node that stalled has its offset
      advanced by its own window width (a deterministic slide deeper
      into the SAME already-computed ranking); a node that found
      something new resets to offset 0 (fresh duals next round make
      re-examining from the top the right default again).
    - The stopping condition changed from "n_new == 0" to "n_new == 0
      AND every node's window has genuinely reached the tail of its
      full candidate list" (`_is_window_saturated`, checked across all n
      nodes). A node whose window hasn't yet been slid through its whole
      candidate list is NOT treated as exhausted, so the run keeps going
      and gives it a materially different (deeper) window on its very
      next attempt -- not a repeat of an already-known-empty search.
  This directly targets "stops after not so many iterations" without
  touching exploration_percent, candidate-selection's exploration slots,
  or introducing any randomness.
* A real bug was caught IN THIS FIX ITSELF while testing it: novelty
  ("did this node's pricing find anything new") was originally checked
  BEFORE the improving-cost filter, not after -- meaning a candidate
  window that slid to new territory would almost always produce
  structurally different node sequences (since the actual candidates
  differ), even when every one of them failed the improving-cost filter
  and never reached the pool. That falsely triggered "found_new = True",
  resetting the offset back to 0 immediately -- undoing the slide and
  re-stalling on the very next call, so windows never actually
  progressed. Fixed by checking novelty only on segments that survive
  the filter and are actually appended to the candidate pool.
* HONEST EMPIRICAL RESULT, tested directly (not just asserted): on
  several small-to-medium synthetic instances (n=14-25, qubit_count=2-3),
  the fix reliably (a) no longer stops at the first n_new==0 iteration --
  confirmed via direct logging showing iteration_log's `fully_saturated`
  staying False well past where the OLD rule would have exited, and the
  run correctly continuing until genuinely exhausted, and (b) in direct
  side-by-side comparisons (same instance, same seed, capped at the OLD
  stopping iteration vs. run to the NEW fully-saturated stopping point),
  final tour cost was IDENTICAL in every test case tried so far -- the
  deeper, previously-unexamined parts of the ranking did not contain a
  better column in these particular (small) instances. This is a
  genuinely different, weaker result than "this fixes the worse-than-
  Amazon results": the STOPPING LOGIC bug is real and fixed (the run no
  longer exits on an unproven claim), but whether that produces BETTER
  tours depends on whether better columns are actually hiding past the
  old narrow window, which these small test cases didn't happen to
  contain. The effect is expected to matter more on real ALMRRC-scale
  routes (100-250+ stops), where a qubit_count=2-4 window is a much
  smaller fraction of the full candidate pool -- but this has NOT been
  tested against real data in this session (only small synthetic
  instances, for runtime reasons) and should be verified before treating
  it as a solved quality problem.
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
                                     duals=None, global_max_dist=None, window_offset=0):
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

    `window_offset` (FIX, see run_cg_hybrid_lrwsqaoa_sub's FIX LOG entry
    "PREMATURE CONVERGENCE"): slices the dual-aware ranking as
    `ranked[window_offset : window_offset + k]` instead of always
    `ranked[:k]`. With window_offset always 0 (the previous behavior),
    once duals stabilize between iterations, the identical top-k window
    gets re-proposed every time, re-derives the identical truncations,
    finds them already pooled, and the run reports "converged" -- while
    never having looked at the (k+1)-th, 10th, or 50th-ranked candidate
    at all. run_cg_hybrid_lrwsqaoa_sub tracks a persistent per-node
    offset across iterations and advances it (deterministically, no
    randomness) whenever a node's current window fails to yield
    anything new, so a stalled node's NEXT attempt looks further down
    the same ranking instead of retrying the identical window forever.
    Deliberately independent of exploration_percent, which stays exactly
    as it already behaves.

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
        # Clamp so a saturated offset re-uses the tail window rather than
        # slicing past the end into an empty/short list.
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
    """True once curr_node's ranking window has already reached the tail
    of its available candidate pool -- i.e. sliding it further would be a
    no-op (see _dual_aware_nearest_and_explore's clamping). Used by
    run_cg_hybrid_lrwsqaoa_sub to distinguish "genuinely exhausted the
    full candidate list for every node" from "just this iteration's
    narrow window found nothing" when deciding whether to actually stop."""
    n = matrix.shape[0]
    others_count = n - 1 - len(exclude)
    if others_count <= 0:
        return True
    max_offset = max(0, others_count - k)
    return window_offset >= max_offset


def _nearest_and_explore_candidates(curr_node, exclude, matrix, k, exploration_percent, rng,
                                     duals=None, global_max_dist=None, window_offset=0):
    nearest, explore = _dual_aware_nearest_and_explore(
        curr_node, exclude, matrix, k, exploration_percent, rng, duals, global_max_dist, window_offset
    )
    return nearest + explore


def _two_opt_open_tsp(tour, matrix, max_iter=None):
    """
    Same style of open-TSP 2-opt local search used in
    algo_hybrid_LRWSQAOA.py's run_algo_hybrid_2_5 (that file's copy was
    already fixed the same way -- this brings this file's independent
    copy in line with it, per explicit request).

    FIX (this revision): `max_iter` used to default to a flat 100 --
    which is a cap on the TOTAL number of improving swaps applied (each
    pass here applies at most one swap before restarting the O(n^2) scan
    from the top), not "100 full passes". For real Amazon routes
    (100-250+ stops), reaching a true 2-opt local optimum from a
    nontrivial starting tour (here, the greedily-concatenated segment
    order -- see _concatenate_segments) can plausibly need several
    hundred to well over a thousand swaps; a fixed 100 regardless of
    route size very likely left larger routes well short of local
    2-opt-optimality, with genuine leftover crossings a fully-converged
    2-opt would have removed. Default is now `max(100, 50 * n)` --
    scales with route size (e.g. 12,500 for a 250-node route) instead of
    a flat constant. Pass an explicit int to override.
    """
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
# Master problem (PuLP set-partitioning LP / ILP)
# =====================================================================

# Cached across calls -- see FIX LOG. `None` = not yet probed;
# afterwards either "coin" or "cbc". Probing solver availability
# (spawning/checking for a binary) is NOT something to do on every one
# of the up-to-(n_iterations+1) master solves in a single run.
_CACHED_SOLVER_CLASS = None


def _make_var(prob, name, low, high, cat):
    """
    PuLP 4.0 deprecates constructing LpVariable directly in favor of
    prob.add_variable(...), which attaches the variable to the problem
    at creation time. Version-safe: uses the new method when available,
    falls back to the old constructor on PuLP versions that don't have
    add_variable yet.
    """
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, lowBound=low, upBound=high, cat=cat)
    return pulp.LpVariable(name, lowBound=low, upBound=high, cat=cat)


def _get_constraint(prob, name):
    """
    PuLP 4.0 deprecates dict-style prob.constraints[name] access in favor
    of prob.get_constraint_by_name(name) (per that version's own
    deprecation message). Version-safe: uses the new method when
    available, falls back to the old dict access on older PuLP. See FIX
    LOG above for why this one specifically mattered for performance,
    not just warning noise.
    """
    if hasattr(prob, "get_constraint_by_name"):
        return prob.get_constraint_by_name(name)
    return prob.constraints[name]


def _make_master_solver(time_limit):
    """
    PuLP 4.0 deprecates PULP_CBC_CMD in favor of COIN_CMD (per that
    version's own deprecation message: "Install CBC with
    `pip install pulp[cbc]` and use COIN_CMD instead"). COIN_CMD needs
    the `pulp[cbc]` extra actually installed to find a CBC binary, which
    may not be true in every environment this runs in -- so this probes
    COIN_CMD's availability ONCE per process (cached in
    _CACHED_SOLVER_CLASS, not re-probed on every master solve -- see FIX
    LOG) and falls back to PULP_CBC_CMD, with its own deprecation
    warning locally suppressed since the fallback here is deliberate.
    """
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
    x = [_make_var(prob, f"x_{i}", 0, 1, cat) for i in range(len(columns))]

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
    global_max_dist=None, window_offset=None, known_sequences=None,
):
    """
    `window_offset`: dict {node: int}, the CURRENT per-node ranking-window
    offset (see _dual_aware_nearest_and_explore's docstring and
    run_cg_hybrid_lrwsqaoa_sub's "PREMATURE CONVERGENCE" fix). Read-only
    here -- treated as {} (all offsets 0) if None.

    `known_sequences`: set of node-sequence tuples already in the pool
    BEFORE this call, used to detect whether a given node's pricing this
    round actually found anything genuinely new (not just re-derived an
    already-pooled column). If None, novelty tracking is skipped (offsets
    passed through unchanged) -- used for the very first iteration, where
    the pool is just singletons and nothing meaningful has stalled yet.

    Returns (priced_columns, next_window_offset): `next_window_offset` is
    a NEW dict, one entry per node actually priced this call -- the
    offset each node's NEXT pricing attempt should use. A node whose
    truncations included at least one genuinely new sequence resets to 0
    (duals will likely have moved, so re-examining from the top under
    fresh duals is the right default); a node whose window stalled
    (nothing new) advances by n_nearest, deterministically, so its next
    attempt looks further down the same ranking instead of retrying the
    identical window.
    """
    n = matrix.shape[0]
    start_nodes = list(range(n))
    if max_pricing_nodes is not None and max_pricing_nodes < n:
        start_nodes = list(rng.choice(n, size=max_pricing_nodes, replace=False))

    window_offset = window_offset or {}
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
    next_window_offset = {}

    for curr_node in start_nodes:
        exclude = {depot_idx} if curr_node != depot_idx else set()
        k_batch = min(qubit_count, n - 1 - len(exclude))
        if k_batch <= 0:
            continue

        offset_here = window_offset.get(curr_node, 0) if apply_dual_candidate_filter else 0
        candidates = _nearest_and_explore_candidates(
            curr_node, exclude, matrix, k_batch, exploration_percent, rng,
            duals=candidate_duals, global_max_dist=global_max_dist, window_offset=offset_here,
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
        found_new = known_sequences is None  # first iteration: nothing to compare against yet
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
            # Novelty is checked AFTER the improving-cost filter, not before
            # it (a real bug this fix caught during its own testing -- see
            # "PREMATURE CONVERGENCE" FIX LOG): a candidate window that
            # slides to new territory will almost always produce
            # structurally different node SEQUENCES than before, even when
            # every one of them fails the improving-cost filter and never
            # actually reaches the pool. Checking novelty before the filter
            # meant `found_new` was true almost every iteration regardless
            # of whether anything useful was found, which reset the offset
            # back to 0 immediately -- undoing the slide and re-stalling on
            # the very next call instead of ever making progress deeper
            # into the ranking.
            if known_sequences is not None and tuple(seg) not in known_sequences:
                found_new = True

        if apply_dual_candidate_filter:
            if found_new:
                next_window_offset[curr_node] = 0  # fresh duals next round -> re-examine from the top
            else:
                next_window_offset[curr_node] = offset_here + k_batch  # slide past this stalled window

    return priced, next_window_offset


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
    # Persistent per-node ranking-window offset -- see
    # _dual_aware_nearest_and_explore's docstring and the "PREMATURE
    # CONVERGENCE" FIX LOG entry below. {} = every node starts at
    # offset 0 (top of its reduced-cost ranking), same as before this fix.
    window_offset = {}

    for it in range(1, n_iterations + 1):
        status_lp, _, _, duals = _solve_master(pool, node_ids, relaxation=True, time_limit=time_limit)
        if status_lp not in ("Optimal", "Not Solved"):
            warnings.warn(f"Iteration {it}: LP relaxation status was '{status_lp}' (expected 'Optimal').")

        known_sequences = {tuple(col["nodes"]) for col in pool} if it >= 2 else None
        priced_columns, next_window_offset = _generate_priced_columns(
            matrix, depot_idx, duals, qubit_count, exploration_percent, xy_mixer,
            only_improving_columns, max_pricing_nodes, rng,
            apply_dual_candidate_filter=(it >= 2), global_max_dist=global_max_dist,
            window_offset=window_offset, known_sequences=known_sequences,
        )
        window_offset = next_window_offset

        pool_before = len(pool)
        pool = _dedupe_columns(pool + priced_columns)
        n_new = len(pool) - pool_before

        # Whether EVERY node's ranking window has genuinely run out of
        # room to slide further (see _is_window_saturated) -- i.e. every
        # pricing start has now had its FULL candidate list examined at
        # some point, not just its original top-k. Only meaningful once
        # apply_dual_candidate_filter has kicked in (it >= 2).
        fully_saturated = it >= 2 and all(
            _is_window_saturated(
                node, {depot_idx} if node != depot_idx else set(), matrix,
                min(qubit_count, n - 1 - (1 if node != depot_idx else 0)),
                window_offset.get(node, 0),
            )
            for node in range(n)
        )

        iteration_log.append({
            "iteration": it,
            "lp_status": status_lp,
            "num_priced": len(priced_columns),
            "num_new_columns": n_new,
            "pool_size": len(pool),
            "fully_saturated": fully_saturated,
        })

        if n_new == 0 and (it < 2 or fully_saturated):
            # FIX (see "PREMATURE CONVERGENCE" in the module FIX LOG):
            # only declare convergence once every node's ranking window
            # has actually been slid through its full candidate list --
            # not merely because THIS iteration's (possibly narrow, or
            # previously-tried) window happened to find nothing new.
            # window_offset advancing on stall (see _generate_priced_
            # columns) means a node that stalls here will use a DEEPER
            # window next call -- so if any node isn't yet saturated,
            # continuing is a genuine, still-productive next attempt, not
            # a repeat of an already-exhausted search.
            break

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
