"""
visualize_step_by_step_CG_BELLMAN_REAL_DATA.py

Bellman-Ford counterpart of visualize_step_by_step_CG_REAL_DATA.py:
runs Column Generation step by step against ONE real Amazon Last-Mile
route, same frame-by-frame structure and the same graphs-only pricing
frame (no candidate/truncation tables -- see that file's own module
docstring for why), but the pricing subproblem is solved EXACTLY via
Bellman-Ford/Held-Karp (cg_hybrid_bellmanford_sub.solve_bellman_ford_
subtour) instead of the heuristic QAOA circuit.

--------------------------------------------------------------------------
WHAT'S DIFFERENT FROM visualize_step_by_step_CG_REAL_DATA.py, AND WHY
--------------------------------------------------------------------------
* Import source: `cg_hybrid_bellmanford_sub` instead of
  `cg_hybrid_lrwsqaoa_sub` / `algo_hybrid_LRWSQAOA` -- no Qiskit
  dependency at all, since there's no quantum circuit here.
* `qubit_count` -> `window_size` throughout (matches
  cg_hybrid_bellmanford_sub.py's own naming; nothing here is a qubit).
* No `xy_mixer` parameter -- nothing to mix.
* `bf_max_k` (default 14): Bellman-Ford/Held-Karp is EXACT but
  exponential (O(2^k * k^2) -- see cg_hybrid_bellmanford_sub.py's own
  module docstring for measured runtimes). This diagnostic calls the
  solver once per (node, iteration) pair -- potentially hundreds of
  calls per run -- so its default window_size and bf_max_k are kept
  modest (4 and 14 respectively) compared to the dedicated scaling
  experiment (experiment_bf_qubit_scaling.py), which is built
  specifically to push k higher with far fewer total calls. Raise
  bf_max_k explicitly if you want a deeper sweep here and have budgeted
  for the runtime.
* Pricing-window sliding fix carried over: cg_hybrid_bellmanford_sub.py's
  premature-convergence fix (a per-node ranking-window offset that
  slides deterministically instead of always reading the top of the
  reduced-cost ranking -- see that file's own "PREMATURE CONVERGENCE"
  FIX LOG entry) is reproduced in this driver's loop too, since this
  file reimplements its own iteration loop (for per-node frame
  rendering control) rather than calling run_cg_hybrid_bellmanford_sub()
  directly. Omitting it here would have silently reintroduced the exact
  bug that fix exists to prevent, just in a second copy of the loop.
* Pricing frame title/labels say "Bellman-Ford sub-tour (exact)" rather
  than "QAOA sub-tour" -- and unlike QAOA, this solver either succeeds
  with the true optimum for the given candidates or raises (if
  window_size > bf_max_k for that call); there is no "weak solve"
  failure mode to distinguish, only "did we skip this node because the
  window was too large."

Everything else -- the graphs-only pricing frame design, the local
_real_data replacements for the final-master/concatenation/2-opt frames
(segment coloring, order-position labels, on-figure and saved path
text), the zoom/padding fix for the pricing map, the arrow-sizing fix --
is identical in spirit and mostly identical in code to
visualize_step_by_step_CG_REAL_DATA.py; see that file's own FIX LOG for
the history of why each of those exists. visualize_step_by_step_CG.py
itself remains untouched, same as before.
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
import cg_hybrid_bellmanford_sub as cg
from cg_hybrid_bellmanford_sub import ITERATION_CG, solve_bellman_ford_subtour
from plot_publication import _style_axes, _plot_directional_route, _mark_start_end, _save_all_formats, COLOR_TEXT

# Reused UNMODIFIED from visualize_step_by_step_CG.py -- these are all
# already graphs (bar chart, line charts, histogram), no tables, so
# nothing about them needed to change, same as the QAOA real-data file.
from visualize_step_by_step_CG import (
    _frame_duals_snapshot,
    _frame_pool_growth,
    _frame_dual_evolution,
    _frame_reduced_cost_histogram,
)
# _frame_final_master, _frame_concatenation, _frame_two_opt_before_after
# are NOT imported from visualize_step_by_step_CG.py -- local _real_data
# replacements below (see module docstring).

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


# =====================================================================
# Shared helper: print AND save a node path (same as the QAOA real-data
# file -- see that file's module docstring for the rationale).
# =====================================================================

def _print_and_save_path(nodes, label, output_dir, txt_filename, cost=None):
    header = label if cost is None else f"{label}  (cost={cost:.2f})"
    path_str = " -> ".join(str(n) for n in nodes)
    text = f"{header}\n  {path_str}\n"
    print(text)
    with open(os.path.join(output_dir, txt_filename), "a") as f:
        f.write(text + "\n")
    return text


def _annotate_order_sparse(ax, coords, path_nodes, max_labels=30, fontsize=7.5):
    """Same as the QAOA real-data file's function of the same name --
    labels points along `path_nodes` with their VISIT ORDER, not raw
    node id."""
    n = len(path_nodes)
    if n == 0:
        return
    if n <= max_labels:
        chosen_positions = list(range(n))
    else:
        step = max(1, (n - 2) // max(max_labels - 2, 1))
        chosen_positions = sorted(set([0, n - 1] + list(range(0, n, step))))
    for pos in chosen_positions:
        node = path_nodes[pos]
        tag = "START" if pos == 0 else ("END" if pos == n - 1 else f"#{pos}")
        ax.annotate(tag, (coords[node, 0], coords[node, 1]), fontsize=fontsize,
                    color=COLOR_TEXT, xytext=(4, 4), textcoords="offset points", zorder=7,
                    fontweight="bold" if pos in (0, n - 1) else "normal")


# =====================================================================
# Pricing frame: graphs only, no tables -- Bellman-Ford version
# =====================================================================

def _frame_pricing_map_only(frame_idx, coords, matrix, curr_node, nearest, explore, full_nodes,
                             iteration, output_dir, formats, full_coord_span):
    """
    Same zoom/padding/arrow-sizing fixes as the QAOA real-data file's
    function of the same name (see that file's FIX LOG for the full
    history of each). Only the title differs: "Bellman-Ford sub-tour
    (exact)" instead of "QAOA sub-tour", since this solver is exact for
    whatever candidates it's given, not a variational heuristic.
    """
    fig, ax = plt.subplots(figsize=(9.5, 8.2))

    ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=30, zorder=1)

    full_path_coords = coords[full_nodes]
    _plot_directional_route(ax, full_path_coords, "#8c1c13", linewidth=1.9, zorder=4)

    _mark_start_end(ax, full_path_coords, "#8c1c13", zorder=6)
    ax.scatter(*coords[curr_node], c="#1a1a1a", s=90, marker="*", zorder=7,
               edgecolors="white", linewidths=0.9, label=f"Start node {curr_node}")

    if nearest:
        nc = coords[nearest]
        ax.scatter(nc[:, 0], nc[:, 1], c="#3b6fa0", s=30, marker="o", zorder=5, label="Nearest candidate")
    if explore:
        ec = coords[explore]
        ax.scatter(ec[:, 0], ec[:, 1], c="#e07b1a", s=30, marker="^", zorder=5, label="Exploration candidate")

    _annotate_order_sparse(ax, coords, full_nodes, max_labels=len(full_nodes))

    ax.set_title(f"Bellman-Ford sub-tour (exact) from node {curr_node}"
                 f"{' (dual-filtered)' if iteration >= 2 else ''}", fontsize=12.5)
    ax.legend(loc="best", fontsize=9)

    # _style_axes() BEFORE the explicit zoom below -- see the QAOA
    # real-data file's FIX LOG for why the order matters (its
    # adjustable="datalim" would otherwise silently undo the zoom).
    _style_axes(ax)

    focus_idx = [curr_node] + nearest + explore
    focus_coords = coords[focus_idx]
    xmin, xmax = focus_coords[:, 0].min(), focus_coords[:, 0].max()
    ymin, ymax = focus_coords[:, 1].min(), focus_coords[:, 1].max()
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    pad = max(span * 0.6, full_coord_span * 0.03)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = span / 2 + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")

    path_str = " \u2192 ".join(str(x) for x in full_nodes)
    ax.text(0.02, 0.02, f"Path: {path_str}", transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left", color=COLOR_TEXT,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))

    fig.suptitle(f"Step 2 \u2014 Pricing Subproblem (Bellman-Ford): node {curr_node}, iteration {iteration}",
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_all_formats(
        fig, os.path.join(output_dir, f"02_{frame_idx:03d}_iter{iteration}_node{curr_node}"), formats
    )
    plt.close(fig)


# =====================================================================
# Final master / concatenation / 2-opt frames -- identical in structure
# to the QAOA real-data file's _real_data replacements (only the
# module-level color accents differ slightly, matching this file's
# Bellman-Ford red/orange palette rather than QAOA's purple/teal).
# =====================================================================

_SEGMENT_COLORS = plt.get_cmap("tab20").colors


def _frame_final_master(coords, depot_idx, selected_columns, pool_size, output_dir, formats):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=22, zorder=1)

    txt_filename = "paths_final_master.txt"
    open(os.path.join(output_dir, txt_filename), "w").close()
    _print_and_save_path(
        [], f"=== Final Master Problem: {len(selected_columns)} segment(s) selected from a pool of {pool_size} ===",
        output_dir, txt_filename,
    )

    legend_handles = []
    for i, col in enumerate(selected_columns):
        color = _SEGMENT_COLORS[i % len(_SEGMENT_COLORS)]
        nodes = col["nodes"]
        path_coords = coords[nodes]
        _plot_directional_route(ax, path_coords, color, linewidth=2.2, zorder=3)
        _mark_start_end(ax, path_coords, color, zorder=5)
        if i < 20:
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=2.5, label=f"Seg {i}: {len(nodes)} node(s)")
            )
        _print_and_save_path(nodes, f"Segment {i}", output_dir, txt_filename, cost=col.get("cost"))

    if len(selected_columns) > 20:
        legend_handles.append(plt.Line2D([0], [0], color="none",
                                          label=f"(+{len(selected_columns) - 20} more -- see {txt_filename})"))

    ax.scatter(*coords[depot_idx], c="#C1272D", s=260, marker="D", edgecolors="white",
               linewidths=1.4, zorder=8, label="Depot")
    ax.set_title(f"Step 3 \u2014 Final Master Problem: {len(selected_columns)} segments selected "
                 f"(pool size {pool_size})", fontsize=13)
    ax.legend(handles=legend_handles + [plt.Line2D([0], [0], marker="D", color="none",
               markerfacecolor="#C1272D", markeredgecolor="white", markersize=10, label="Depot")],
               loc="best", fontsize=7.5, ncol=2 if len(legend_handles) > 10 else 1)
    _style_axes(ax)
    fig.tight_layout()
    _save_all_formats(fig, os.path.join(output_dir, "03_final_master"), formats)
    plt.close(fig)
    print(f"[saved segment paths -> {os.path.join(output_dir, txt_filename)}]")


def _frame_concatenation(coords, depot_idx, selected_columns, raw_tour, matrix, output_dir, formats):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=22, zorder=1)

    path_coords = coords[raw_tour]
    _plot_directional_route(ax, path_coords, "#8c1c13", linewidth=2.2, zorder=3)
    _mark_start_end(ax, path_coords, "#8c1c13", zorder=6)
    _annotate_order_sparse(ax, coords, raw_tour, max_labels=30)

    ax.scatter(*coords[depot_idx], c="#C1272D", s=260, marker="D", edgecolors="white",
               linewidths=1.4, zorder=8, label="Depot")
    raw_cost = cg._open_path_cost(raw_tour, matrix)
    ax.set_title(f"Step 4 \u2014 Concatenated Route (pre-2-opt): {len(raw_tour)} stops, cost {raw_cost:.1f}",
                 fontsize=13)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="#8c1c13",
                   markeredgecolor="white", markersize=10, label="Start"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#8c1c13",
                   markeredgecolor="white", markersize=9, label="End"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="#C1272D",
                   markeredgecolor="white", markersize=10, label="Depot"),
    ], loc="best", fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    _save_all_formats(fig, os.path.join(output_dir, "04_concatenation"), formats)
    plt.close(fig)

    txt_filename = "paths_route.txt"
    _print_and_save_path(raw_tour, "Concatenated route (pre-2-opt)", output_dir, txt_filename, cost=raw_cost)
    print(f"[saved route path -> {os.path.join(output_dir, txt_filename)}]")


def _frame_two_opt_before_after(coords, depot_idx, raw_tour, final_tour, raw_cost, final_cost, output_dir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5))

    for ax, tour, cost, title in [
        (axes[0], raw_tour, raw_cost, "Before 2-opt"),
        (axes[1], final_tour, final_cost, "After 2-opt"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=22, zorder=1)
        path_coords = coords[tour]
        _plot_directional_route(ax, path_coords, "#8c1c13", linewidth=2.0, zorder=3)
        _mark_start_end(ax, path_coords, "#8c1c13", zorder=6)
        _annotate_order_sparse(ax, coords, tour, max_labels=25)
        ax.scatter(*coords[depot_idx], c="#C1272D", s=220, marker="D", edgecolors="white",
                   linewidths=1.3, zorder=8)
        ax.set_title(f"{title}: {len(tour)} stops, cost {cost:.1f}", fontsize=12.5)
        _style_axes(ax)

    pct = 100 * (raw_cost - final_cost) / raw_cost if raw_cost else 0.0
    fig.suptitle(f"Step 5 \u2014 2-opt Polish ({pct:+.1f}% cost change)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_all_formats(fig, os.path.join(output_dir, "05_two_opt_before_after"), formats)
    plt.close(fig)

    txt_filename = "paths_route.txt"
    _print_and_save_path(final_tour, "FINAL route (after 2-opt) -- this is the delivered tour",
                          output_dir, txt_filename, cost=final_cost)
    print(f"[saved final route path -> {os.path.join(output_dir, txt_filename)}]")


# =====================================================================
# Driver -- same algorithmic logic as
# cg_hybrid_bellmanford_sub.run_cg_hybrid_bellmanford_sub(), including
# its sliding-window premature-convergence fix (see module docstring),
# reimplemented here only because per-node frame rendering needs
# control the packaged function doesn't expose.
# =====================================================================

def visualize_cg_stepwise_execution_bellman_real_data(
    data,
    window_size=4,
    exploration_percent=0.0,
    only_improving_columns=True,
    n_iterations=3,
    detail_iterations=None,
    max_detail_nodes=None,
    seed=101,
    output_dir="cg_visualizations_bellman_real_data",
    formats=("png",),
    bf_max_k=14,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    matrix = data["matrix"]
    coords = data["coords"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)
    n_iterations = max(1, n_iterations)
    global_max_dist = float(matrix.max()) if n > 1 else 0.0

    full_coord_span = max(
        float(coords[:, 0].max() - coords[:, 0].min()),
        float(coords[:, 1].max() - coords[:, 1].min()),
        1e-6,
    )

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
    window_offset = {}  # per-node sliding ranking-window offset -- see module docstring

    print(f"Running {n_iterations} pricing iteration(s) (detail frames at iterations "
          f"{sorted(detail_iterations)}, nodes {sorted(detail_nodes)}; graphs only, no tables; "
          f"Bellman-Ford exact pricing, bf_max_k={bf_max_k})...")

    for it in range(1, n_iterations + 1):
        status_lp, _, _, duals = cg._solve_master(pool, list(range(n)), relaxation=True)
        dual_history.append(dict(duals))
        render_detail_this_iter = it in detail_iterations

        if render_detail_this_iter:
            _frame_duals_snapshot(depot_idx, duals, n, it, len(pool), output_dir, formats)

        priced_this_iter = []
        apply_dual_candidate_filter = (it >= 2)
        known_sequences = {tuple(col["nodes"]) for col in pool} if it >= 2 else None
        next_window_offset = {}

        if apply_dual_candidate_filter:
            duals_vector = np.array([duals.get(j, 0.0) for j in range(n)])
            bf_matrix = matrix - duals_vector[np.newaxis, :]
        else:
            bf_matrix = matrix

        for curr_node in range(n):
            exclude = {depot_idx} if curr_node != depot_idx else set()
            k_batch = min(window_size, n - 1 - len(exclude))
            if k_batch <= 0:
                continue
            if k_batch > bf_max_k:
                k_batch = bf_max_k  # see module docstring re: bf_max_k default

            offset_here = window_offset.get(curr_node, 0) if apply_dual_candidate_filter else 0
            nearest, explore = cg._dual_aware_nearest_and_explore(
                curr_node, exclude, matrix, k_batch, exploration_percent, rng,
                duals=(duals if apply_dual_candidate_filter else None),
                global_max_dist=global_max_dist, window_offset=offset_here,
            )
            candidates = nearest + explore
            want_detail = render_detail_this_iter and curr_node in detail_nodes

            if not candidates:
                if want_detail:
                    pricing_frame_counter += 1
                    _frame_pricing_map_only(pricing_frame_counter, coords, matrix, curr_node,
                                             nearest, explore, [curr_node], it, output_dir, formats,
                                             full_coord_span)
                continue

            try:
                subtour, _ = solve_bellman_ford_subtour(curr_node, candidates, bf_matrix, max_k=bf_max_k)
            except ValueError as e:
                warnings.warn(f"Skipping node {curr_node} at iteration {it}: {e}")
                continue
            if not subtour:
                if want_detail:
                    pricing_frame_counter += 1
                    _frame_pricing_map_only(pricing_frame_counter, coords, matrix, curr_node,
                                             nearest, explore, [curr_node], it, output_dir, formats,
                                             full_coord_span)
                continue
            full_nodes = [curr_node] + subtour

            found_new = known_sequences is None
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
                    if known_sequences is not None and tuple(seg) not in known_sequences:
                        found_new = True

            if apply_dual_candidate_filter:
                next_window_offset[curr_node] = 0 if found_new else offset_here + k_batch

            if want_detail:
                pricing_frame_counter += 1
                _frame_pricing_map_only(pricing_frame_counter, coords, matrix, curr_node,
                                         nearest, explore, full_nodes, it, output_dir, formats,
                                         full_coord_span)

        window_offset = next_window_offset
        pool_before = len(pool)
        pool = cg._dedupe_columns(pool + priced_this_iter)
        n_new = len(pool) - pool_before

        fully_saturated = it >= 2 and all(
            cg._is_window_saturated(
                node, {depot_idx} if node != depot_idx else set(), matrix,
                min(window_size, n - 1 - (1 if node != depot_idx else 0), bf_max_k),
                window_offset.get(node, 0),
            )
            for node in range(n)
        )

        iteration_log.append({
            "iteration": it, "lp_status": status_lp, "num_priced": len(priced_this_iter),
            "num_new_columns": n_new, "pool_size": len(pool), "fully_saturated": fully_saturated,
        })
        print(f"  iteration {it}: priced {len(priced_this_iter)}, new {n_new}, pool size {len(pool)}, "
              f"fully_saturated={fully_saturated}")

        if n_new == 0 and (it < 2 or fully_saturated):
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
        "final_tour": final_tour, "final_cost": final_cost, "raw_tour": raw_tour, "raw_cost": raw_cost,
        "iteration_log": iteration_log, "dual_history": dual_history,
        "num_pool_columns": len(full_pool), "num_segments_selected": len(selected_columns),
    }


# =====================================================================
# Real-data loading -- identical to the QAOA real-data file's function
# of the same name.
# =====================================================================

def load_one_real_route(data_dir, route_id=None, seed=2026):
    if not os.path.exists(data_dir) and os.path.exists("./data"):
        data_dir = "./data"

    loader = AmazonDataLoader(data_dir=data_dir)
    if not loader.travel_times:
        raise FileNotFoundError(
            f"No route data found in '{data_dir}'. Ensure travel_times.json is available."
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
        description="Step-by-step CG diagnostic visualization on one real Amazon route, using EXACT "
                    "Bellman-Ford/Held-Karp pricing instead of QAOA (graphs only, no tables)."
    )
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training")
    parser.add_argument("--route-id", type=str, default=None,
                         help="Specific route ID to use. If omitted, one is picked automatically (seeded).")
    parser.add_argument("--window-size", type=int, default=4,
                         help="Candidate-window size k for each pricing call (default 4).")
    parser.add_argument("--bf-max-k", type=int, default=14,
                         help="Safety cap on k for the exact solver (default 14 -- see module docstring "
                              "for why this is lower here than in the dedicated scaling experiment).")
    parser.add_argument("--exploration-percent", type=float, default=0.1)
    parser.add_argument("--no-only-improving-columns", action="store_true")
    parser.add_argument("--n-iterations", type=int, default=3,
                         help=f"Pricing iterations (default 3, deliberately lower than the global "
                              f"ITERATION_CG={ITERATION_CG} default).")
    parser.add_argument("--max-detail-nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default="./cg_visualizations_bellman_real_data")
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
        f"  Each point-iteration involves one EXACT Bellman-Ford solve (window_size={args.window_size}, "
        f"bf_max_k={args.bf_max_k}) -- see cg_hybrid_bellmanford_sub.py's module docstring for measured\n"
        f"  per-call runtimes at various k. Use --max-detail-nodes and/or --n-iterations to scope this\n"
        f"  down if needed.\n"
    )
    if not args.yes:
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            resp = "y"
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    result = visualize_cg_stepwise_execution_bellman_real_data(
        data,
        window_size=args.window_size,
        exploration_percent=args.exploration_percent,
        only_improving_columns=not args.no_only_improving_columns,
        n_iterations=args.n_iterations,
        max_detail_nodes=args.max_detail_nodes,
        seed=args.seed,
        output_dir=args.output_dir,
        bf_max_k=args.bf_max_k,
    )

    print(f"\nDone. Final tour cost: {result['final_cost']:.2f} "
          f"(raw pre-2opt: {result['raw_cost']:.2f}). "
          f"Frames saved under '{args.output_dir}/'")


if __name__ == "__main__":
    main()
