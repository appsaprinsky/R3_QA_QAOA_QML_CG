"""
visualize_step_by_step_CG_REAL_DATA.py

Runs the CG algorithm step by step against ONE real Amazon Last-Mile
route, same overall structure as visualize_step_by_step_CG.py, but with
one deliberate difference: the pricing-subproblem frames here show ONLY
the map/graph -- no candidate-selection table, no truncation table.

--------------------------------------------------------------------------
WHY THIS FILE DIVERGES FROM visualize_step_by_step_CG.py, AND HOW
--------------------------------------------------------------------------
visualize_step_by_step_CG.py's pricing frames are a 3-panel figure: map,
candidate-selection economics table, prefix-truncation table. At real
Amazon-route scale (100-250 stops) that table -- even wrapped into
several bounded-height columns -- is still a lot to take in, and isn't
the part that's actually useful to see at a glance for a real route;
the route/candidate geometry is. So this script's pricing frames are
graph-only.

visualize_step_by_step_CG.py itself is NOT modified to get this --
per instruction, it is left exactly as it is, tables included, for
synthetic-data use. Since the table rendering lives inside that file's
_frame_pricing_node() and there's no toggle for it (adding one would
mean editing that file), this script reimplements its own pricing-frame
function (graphs only) and its own driving loop -- but everything else
is imported and reused UNMODIFIED from visualize_step_by_step_CG.py:
_frame_duals_snapshot, _frame_pool_growth, _frame_dual_evolution,
_frame_reduced_cost_histogram, _frame_final_master, _frame_concatenation,
_frame_two_opt_before_after are all already graphs with no tables, so
there was nothing to change about them -- only the one frame type that
had tables needed a replacement. The driving loop's algorithmic logic
(candidate selection, the dual-adjusted QAOA matrix from iteration >= 2,
truncation pricing, pool growth, final master, concatenation, 2-opt) is
copied faithfully from visualize_cg_stepwise_execution() so results
match exactly for the same seed/parameters -- only the pricing-frame
call and the (now unnecessary) per-candidate table bookkeeping differ.
As a side benefit, skipping that bookkeeping also makes this script
faster than the table version at real-route scale.

--------------------------------------------------------------------------
WHY OTHER DEFAULTS ARE DIFFERENT FROM visualize_step_by_step_CG.py
--------------------------------------------------------------------------
Real Amazon routes typically have 100-250 stops, not the 16-40 used in
the synthetic examples. n_iterations defaults to 3 here (not the global
ITERATION_CG=10) since "every point x every iteration" at real-route
scale means many real QAOA solves; a pre-flight estimate is printed
before running so the actual frame count is visible up front rather
than discovered after waiting. max_detail_nodes/detail_iterations
remain available to scope a run down further if needed.
"""

import argparse
import math
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algo_data_loader import AmazonDataLoader
from algo_hybrid_LRWSQAOA import solve_wslr_qaoa_subtour
import cg_hybrid_lrwsqaoa_sub as cg
from cg_hybrid_lrwsqaoa_sub import ITERATION_CG
from plot_publication import _style_axes, _plot_directional_route, _save_all_formats, COLOR_TEXT

# Reused UNMODIFIED from visualize_step_by_step_CG.py -- these are all
# already graphs (bar chart, line charts, histogram, route maps), no
# tables, so nothing about them needed to change.
from visualize_step_by_step_CG import (
    _frame_duals_snapshot,
    _frame_pool_growth,
    _frame_dual_evolution,
    _frame_reduced_cost_histogram,
    _frame_final_master,
    _frame_concatenation,
    _frame_two_opt_before_after,
)

# --- CRITICAL CPU & THERMAL LIMITS (matches run_amazon_experiment.py / run_CG_experiment.py) ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


# =====================================================================
# Pricing frame: graphs only, no tables
# =====================================================================

def _frame_pricing_map_only(frame_idx, coords, matrix, curr_node, nearest, explore, full_nodes,
                             iteration, output_dir, formats):
    fig, ax = plt.subplots(figsize=(9.5, 8.2))

    ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=30, zorder=1)
    if nearest:
        nc = coords[nearest]
        ax.scatter(nc[:, 0], nc[:, 1], c="#3b6fa0", s=130, marker="o", zorder=3,
                   edgecolors="white", linewidths=0.8, label="Nearest candidate")
    if explore:
        ec = coords[explore]
        ax.scatter(ec[:, 0], ec[:, 1], c="#e07b1a", s=130, marker="^", zorder=3,
                   edgecolors="white", linewidths=0.8, label="Exploration candidate")

    full_path_coords = coords[full_nodes]
    _plot_directional_route(ax, full_path_coords, "#5b2d8e", linewidth=1.9, zorder=4)
    ax.scatter(*coords[curr_node], c="#1a1a1a", s=230, marker="*", zorder=6,
               edgecolors="white", linewidths=1.1, label=f"Start node {curr_node}")

    label_nodes = [curr_node] + nearest + explore
    if len(label_nodes) > 40:  # keep the map legible even with a large candidate set
        label_nodes = label_nodes[:40]
    for idx in label_nodes:
        ax.annotate(str(idx), (coords[idx, 0], coords[idx, 1]), fontsize=8,
                    color=COLOR_TEXT, xytext=(3, 3), textcoords="offset points", zorder=7)

    ax.set_title(f"QAOA sub-tour from node {curr_node}"
                 f"{' (dual-filtered)' if iteration >= 2 else ''}", fontsize=12.5)
    ax.legend(loc="best", fontsize=9)
    focus_idx = [curr_node] + nearest + explore
    focus_coords = coords[focus_idx]
    xmin, xmax = focus_coords[:, 0].min(), focus_coords[:, 0].max()
    ymin, ymax = focus_coords[:, 1].min(), focus_coords[:, 1].max()
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    pad = max(span * 0.6, 3.0)  # generous padding + floor so a tight cluster doesn't over-zoom
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = span / 2 + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")  # "box", not "datalim" — keeps these explicit limits
    _style_axes(ax)

    fig.suptitle(f"Step 2 \u2014 Pricing Subproblem: node {curr_node}, iteration {iteration}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_all_formats(
        fig, os.path.join(output_dir, f"02_{frame_idx:03d}_iter{iteration}_node{curr_node}"), formats
    )
    plt.close(fig)


# =====================================================================
# Driver -- same algorithmic logic as visualize_cg_stepwise_execution(),
# reimplemented here only because the pricing-frame call and the
# candidate-table bookkeeping around it differ. See module docstring.
# =====================================================================

def visualize_cg_stepwise_execution_real_data(
    data,
    qubit_count=4,
    exploration_percent=0.0,
    xy_mixer=False,
    only_improving_columns=True,
    n_iterations=3,
    detail_iterations=None,
    max_detail_nodes=None,
    seed=101,
    output_dir="cg_visualizations_real_data",
    formats=("png",),
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    matrix = data["matrix"]
    coords = data["coords"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)
    n_iterations = max(1, n_iterations)
    global_max_dist = float(matrix.max()) if n > 1 else 0.0

    if detail_iterations is None:
        detail_iterations = set(range(1, n_iterations + 1))
    else:
        detail_iterations = set(detail_iterations)

    if max_detail_nodes is None:
        detail_nodes = set(range(n))
    else:
        detail_nodes = set(np.linspace(0, n - 1, num=min(max_detail_nodes, n), dtype=int).tolist())
        detail_nodes.add(depot_idx)

    dual_evolution_sample = set(np.linspace(0, n - 1, num=min(10, n), dtype=int).tolist())
    dual_evolution_sample.add(depot_idx)

    t_start = time.time()
    pool = cg._build_initial_columns(n, matrix, depot_idx)
    dual_history = []
    iteration_log = []
    all_truncations_for_stats = []
    pricing_frame_counter = 0

    print(f"Running {n_iterations} pricing iteration(s) (detail frames at iterations "
          f"{sorted(detail_iterations)}, nodes {sorted(detail_nodes)}; graphs only, no tables)...")

    for it in range(1, n_iterations + 1):
        status_lp, _, _, duals = cg._solve_master(pool, list(range(n)), relaxation=True)
        dual_history.append(dict(duals))
        render_detail_this_iter = it in detail_iterations

        if render_detail_this_iter:
            _frame_duals_snapshot(depot_idx, duals, n, it, len(pool), output_dir, formats)

        priced_this_iter = []
        apply_dual_candidate_filter = (it >= 2)

        if apply_dual_candidate_filter:
            duals_vector = np.array([duals.get(j, 0.0) for j in range(n)])
            qaoa_matrix = matrix - duals_vector[np.newaxis, :]
        else:
            qaoa_matrix = matrix

        for curr_node in range(n):
            exclude = {depot_idx} if curr_node != depot_idx else set()
            k_batch = min(qubit_count, n - 1 - len(exclude))
            if k_batch <= 0:
                continue

            nearest, explore = cg._dual_aware_nearest_and_explore(
                curr_node, exclude, matrix, k_batch, exploration_percent, rng,
                duals=(duals if apply_dual_candidate_filter else None),
                global_max_dist=global_max_dist,
            )
            candidates = nearest + explore
            want_detail = render_detail_this_iter and curr_node in detail_nodes

            if not candidates:
                if want_detail:
                    pricing_frame_counter += 1
                    _frame_pricing_map_only(pricing_frame_counter, coords, matrix, curr_node,
                                             nearest, explore, [curr_node], it, output_dir, formats)
                continue

            subtour = solve_wslr_qaoa_subtour(curr_node, candidates, qaoa_matrix, xy_mixer=xy_mixer)
            if not subtour:
                if want_detail:
                    pricing_frame_counter += 1
                    _frame_pricing_map_only(pricing_frame_counter, coords, matrix, curr_node,
                                             nearest, explore, [curr_node], it, output_dir, formats)
                continue
            full_nodes = [curr_node] + subtour

            for L in range(len(full_nodes), 0, -1):
                seg = full_nodes[:L]
                seg_cost = cg._open_path_cost(seg, matrix)
                dual_sum = sum(duals.get(node, 0.0) for node in seg)
                reduced_cost = seg_cost - dual_sum
                kept = (reduced_cost < -1e-9) or (L == 1)
                record = {"nodes": seg, "cost": seg_cost, "dual_sum": dual_sum,
                          "reduced_cost": reduced_cost, "kept": kept, "start": curr_node}
                all_truncations_for_stats.append(record)
                if (not only_improving_columns) or kept:
                    priced_this_iter.append(record)

            if want_detail:
                pricing_frame_counter += 1
                _frame_pricing_map_only(pricing_frame_counter, coords, matrix, curr_node,
                                         nearest, explore, full_nodes, it, output_dir, formats)

        pool_before = len(pool)
        pool = cg._dedupe_columns(pool + priced_this_iter)
        n_new = len(pool) - pool_before
        iteration_log.append({
            "iteration": it, "lp_status": status_lp, "num_priced": len(priced_this_iter),
            "num_new_columns": n_new, "pool_size": len(pool),
        })
        print(f"  iteration {it}: priced {len(priced_this_iter)}, new {n_new}, pool size {len(pool)}")

        if n_new == 0:
            print(f"  Converged after {it} iteration(s) -- stopping early.")
            break

    _frame_pool_growth(iteration_log, output_dir, formats)
    _frame_dual_evolution(dual_history, sorted(dual_evolution_sample), depot_idx, output_dir, formats)
    _frame_reduced_cost_histogram(all_truncations_for_stats, output_dir, formats)

    full_pool = pool
    print(f"Final pool size: {len(full_pool)}. Solving final ILP master...")
    status_final, selected_idx, _, _ = cg._solve_master(full_pool, list(range(n)), relaxation=False)
    if status_final == "Optimal" and selected_idx:
        selected_columns = [full_pool[i] for i in selected_idx]
    else:
        warnings.warn(f"Final master status '{status_final}'; using greedy fallback for this visualization.")
        selected_columns = cg._greedy_set_cover_fallback(full_pool, list(range(n)))
    print(f"  status={status_final}, segments selected={len(selected_columns)}")
    _frame_final_master(coords, depot_idx, selected_columns, len(full_pool), output_dir, formats)

    print("Concatenating segments...")
    raw_tour = cg._concatenate_segments(selected_columns, depot_idx, matrix)
    raw_cost = cg._open_path_cost(raw_tour, matrix)
    _frame_concatenation(coords, depot_idx, selected_columns, raw_tour, matrix, output_dir, formats)

    print("Running 2-opt polish...")
    final_tour = cg._two_opt_open_tsp(raw_tour, matrix)
    final_cost = cg._open_path_cost(final_tour, matrix)
    _frame_two_opt_before_after(coords, depot_idx, raw_tour, final_tour, raw_cost, final_cost, output_dir, formats)

    print(f"\nDone in {time.time()-t_start:.2f}s over {len(iteration_log)} iteration(s). "
          f"Raw cost {raw_cost:.2f} -> Final cost {final_cost:.2f} "
          f"({100*(raw_cost-final_cost)/raw_cost:.1f}% from 2-opt). Frames saved under '{output_dir}/'")

    return {
        "final_tour": final_tour,
        "final_cost": final_cost,
        "raw_tour": raw_tour,
        "raw_cost": raw_cost,
        "iteration_log": iteration_log,
        "dual_history": dual_history,
        "num_pool_columns": len(full_pool),
        "num_segments_selected": len(selected_columns),
    }


# =====================================================================
# Real-data loading
# =====================================================================

def load_one_real_route(data_dir, route_id=None, seed=2026):
    """
    Loads exactly one real Amazon route in the same dict shape
    visualize_cg_stepwise_execution_real_data() expects: matrix, coords,
    depot_idx, n_nodes, route_id.
    """
    if not os.path.exists(data_dir) and os.path.exists("./data"):
        data_dir = "./data"

    loader = AmazonDataLoader(data_dir=data_dir)
    if not loader.travel_times:
        raise FileNotFoundError(
            f"No route data found in '{data_dir}'. Ensure travel_times.json is available "
            f"(same data directory used by run_amazon_experiment.py / run_CG_experiment.py)."
        )

    all_route_ids = sorted(loader.travel_times.keys())
    if route_id is None:
        rng = np.random.default_rng(seed)
        route_id = str(rng.choice(all_route_ids))
    elif route_id not in all_route_ids:
        raise ValueError(
            f"route_id '{route_id}' not found in '{data_dir}'. "
            f"{len(all_route_ids)} routes available; pass --route-id with one of them, "
            f"or omit --route-id to pick one automatically."
        )

    extracted = loader.extract_single_route(route_id)
    matrix = np.array(extracted["matrix"])
    coords = np.array(extracted["coords"])

    if coords is None or np.all(coords == 0):
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed)
        coords = mds.fit_transform(matrix)

    return {
        "route_id": extracted.get("route_id", route_id),
        "n_nodes": extracted["n_nodes"],
        "coords": coords,
        "matrix": matrix,
        "depot_idx": extracted.get("depot_idx", 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step CG diagnostic visualization on one real Amazon route (graphs only, no tables)."
    )
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training", help="Dataset directory")
    parser.add_argument("--route-id", type=str, default=None,
                         help="Specific route ID to use. If omitted, one is picked automatically (seeded).")
    parser.add_argument("--qubit-count", type=int, default=4)
    parser.add_argument("--exploration-percent", type=float, default=0.1)
    parser.add_argument("--xy-mixer", action="store_true")
    parser.add_argument("--no-only-improving-columns", action="store_true",
                         help="Disable the reduced-cost filter (keep every priced truncation, not just improving ones).")
    parser.add_argument("--n-iterations", type=int, default=3,
                         help="Pricing iterations for THIS script (default 3, deliberately lower than the "
                              f"global ITERATION_CG={ITERATION_CG} default -- see module docstring for why).")
    parser.add_argument("--max-detail-nodes", type=int, default=None,
                         help="Cap detail frames to this many points per rendered iteration. "
                              "Default None = every point.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default="./cg_visualizations_real_data")
    parser.add_argument("--yes", action="store_true", help="Skip the pre-flight confirmation prompt.")

    args = parser.parse_args()

    print(f"Loading a real Amazon route from '{args.data_dir}'"
          f"{f' (route_id={args.route_id})' if args.route_id else ' (auto-selected)'}...")
    data = load_one_real_route(args.data_dir, route_id=args.route_id, seed=args.seed)
    n = data["n_nodes"]
    print(f"  route_id={data['route_id']}, {n} stops (including depot)")

    detail_node_count = n if args.max_detail_nodes is None else min(args.max_detail_nodes, n)
    est_pricing_frames = detail_node_count * args.n_iterations
    print(
        f"\nPre-flight estimate:\n"
        f"  {detail_node_count} points x {args.n_iterations} iteration(s) = "
        f"{est_pricing_frames} pricing-detail frames (graphs only, no tables) plus ~7 summary frames.\n"
        f"  Each point-iteration involves at least one real QAOA statevector solve --\n"
        f"  this can be slow at real-route scale. Use --max-detail-nodes and/or\n"
        f"  --n-iterations to scope this down if needed.\n"
    )
    if not args.yes:
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            resp = "y"
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    result = visualize_cg_stepwise_execution_real_data(
        data,
        qubit_count=args.qubit_count,
        exploration_percent=args.exploration_percent,
        xy_mixer=args.xy_mixer,
        only_improving_columns=not args.no_only_improving_columns,
        n_iterations=args.n_iterations,
        max_detail_nodes=args.max_detail_nodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"\nDone. Final tour cost: {result['final_cost']:.2f} "
          f"(raw pre-2opt: {result['raw_cost']:.2f}). "
          f"Frames saved under '{args.output_dir}/'")


if __name__ == "__main__":
    main()
