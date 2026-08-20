"""
visualize_step_by_step_CG.py

Step-by-step diagnostic visualizer for the Column-Generation Hybrid
algorithm (cg_hybrid_lrwsqaoa_sub.py). Walks through all `n_iterations`
pricing rounds (default: cg_hybrid_lrwsqaoa_sub.ITERATION_CG), not just a
single snapshot, then the final master solve:

  1. Per-iteration duals -- rendered for EVERY iteration by default
     (see detail_iterations if you want to restrict this).
  2. The pricing subproblem in detail, for EVERY starting point, at
     EVERY rendered iteration by default (see max_detail_nodes if you
     want to restrict this) -- which candidates were considered, what
     path QAOA found, and for every truncation of that path, its cost,
     the sum of duals it covers, its reduced cost, and whether it
     survived the filter. This matches the algorithm's actual structure
     exactly: every point is a starting point at every iteration in
     cg_hybrid_lrwsqaoa_sub.py's _generate_priced_columns, with no
     depletion of the candidate pool across different starting points --
     nothing here is a sample of that process.
  3. Pool growth across all iterations -- pool size and newly-added
     columns per iteration, so you can see when (or whether) pricing
     converges before hitting the iteration cap.
  4. Dual evolution across all iterations, for a small evenly-spaced
     sample of nodes (this chart is a trend summary, not an exhaustive
     per-node account like item 2 above -- plotting every node here
     stops being readable long before n approaches real route sizes) --
     do the shadow prices stabilize, oscillate, or drift?
  5. A reduced-cost histogram across every priced column from every
     iteration.
  6. The final master ILP's selected segments, each drawn in its own
     color.
  7. How those segments get concatenated into one tour -- QAOA-found
     edges (within a segment) are drawn differently from the greedy
     "stitch" edges added between segments.
  8. Before/after cost effect of the closing 2-opt pass.

Does not modify cg_hybrid_lrwsqaoa_sub.py or algo_hybrid_LRWSQAOA.py --
it imports and drives their existing functions directly (including a
few underscore-prefixed internals from cg_hybrid_lrwsqaoa_sub.py, since
this is a companion diagnostic script for that exact module, the same
way visualize_step_by_step.py drives solve_wslr_qaoa_subtour from
algo_hybrid_LRWSQAOA.py without modifying it). ITERATION_CG itself is
imported from cg_hybrid_lrwsqaoa_sub.py rather than redefined here, so
there is one single source of truth for the default iteration count.

Saves frames into 'cg_visualizations/' without GUI blocking.
"""

import math
import os
import time
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from algo_hybrid_LRWSQAOA import solve_wslr_qaoa_subtour
import cg_hybrid_lrwsqaoa_sub as cg
from cg_hybrid_lrwsqaoa_sub import ITERATION_CG
from plot_publication import (
    _style_axes,
    _plot_directional_route,
    _save_all_formats,
    COLOR_DEPOT,
    COLOR_GRID,
    COLOR_TEXT,
)

SEGMENT_CMAP = plt.get_cmap("tab20")
ITERATION_CMAP = plt.get_cmap("viridis")
COLOR_STITCH = "#C1272D"


def generate_random_tsp_data(n_nodes=16, seed=101):
    """Same convention as visualize_step_by_step.py's generator, so both
    scripts can be pointed at directly comparable synthetic instances."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(10, 90, size=(n_nodes, 2))
    coords[0] = [50.0, 50.0]  # Depot
    matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return {"n_nodes": n_nodes, "coords": coords, "matrix": matrix, "depot_idx": 0}


# =====================================================================
# Per-iteration duals snapshot
# =====================================================================

def _frame_duals_snapshot(depot_idx, duals, n, iteration, pool_size, output_dir, formats):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    node_ids = list(range(n))
    values = [duals.get(i, 0.0) for i in node_ids]
    colors = [COLOR_DEPOT if i == depot_idx else "#3b6fa0" for i in node_ids]
    ax.bar(node_ids, values, color=colors, width=0.75, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Node index")
    ax.set_ylabel("Dual price  $\\pi_i$")
    ax.set_title(f"Iteration {iteration} \u2014 LP duals over a pool of {pool_size} columns", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=COLOR_GRID, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"Step 1.{iteration} \u2014 Master LP Duals", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_all_formats(fig, os.path.join(output_dir, f"01_{iteration:02d}_duals"), formats)
    plt.close(fig)


# =====================================================================
# Pricing subproblem detail for a sampled node within a sampled iteration
# =====================================================================

MAX_CANDIDATE_ROWS_PER_COLUMN = 22  # bounds table height regardless of how many candidates exist


def _draw_minitable(ax, rows_chunk, col_labels, cell_formatter, title=None, col_widths=None):
    """Draws one bounded-height table (<= MAX_CANDIDATE_ROWS_PER_COLUMN rows)
    into `ax`. Used to render the candidate table as several side-by-side
    column-groups instead of one arbitrarily long list -- see
    _frame_pricing_node for why."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    if not rows_chunk:
        return
    cell_text, cell_colors = [], []
    for row in rows_chunk:
        text, color = cell_formatter(row)
        cell_text.append(text)
        cell_colors.append([color] * len(col_labels))

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                      cellLoc="center", loc="upper center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2B2B2B")
            cell.set_text_props(color="white", weight="bold")
        elif c == 0:
            cell.set_text_props(ha="left")
        cell.set_edgecolor("#cccccc")
    if title:
        ax.set_title(title, fontsize=9, pad=8)


def _frame_pricing_node(
    frame_idx, coords, matrix, curr_node, nearest, explore, full_nodes,
    truncation_records, candidate_rows, iteration, output_dir, formats,
):
    """
    Three logical panels: map, candidate-selection economics, prefix
    truncations. The candidate panel shows EVERY other node (per the
    "go through all points" requirement) -- but on a real ~100-250 stop
    route that is a lot of rows, and a single vertical table that tall
    was making the WHOLE FIGURE grow to 40-60+ inches tall (figure
    height was scaled linearly with row count), which meant the map
    panel got squeezed to an invisible sliver whenever the image was
    viewed at any normal scale. Fixed here by wrapping the candidate
    list into several side-by-side column-groups of at most
    MAX_CANDIDATE_ROWS_PER_COLUMN rows each, so figure height stays
    bounded regardless of node count -- the map is always a normal,
    visible size. Width grows with candidate count instead (more
    columns), which scales far more usably in an image viewer than
    height did.
    """
    n_cand_cols = max(1, math.ceil(len(candidate_rows) / MAX_CANDIDATE_ROWS_PER_COLUMN)) if candidate_rows else 1
    rows_per_col = MAX_CANDIDATE_ROWS_PER_COLUMN

    # Figure height bounded by the larger of (one candidate column's row
    # count, truncation row count) -- NOT the total candidate count.
    max_rows_for_height = max(min(len(candidate_rows), rows_per_col) if candidate_rows else 1,
                               len(truncation_records), 1)
    fig_height = max(6.6, 1.8 + 0.30 * max_rows_for_height)
    fig_width = 7.6 + 3.1 * n_cand_cols + 6.4

    width_ratios = [1.35] + [1.0] * n_cand_cols + [1.15]
    fig, axes = plt.subplots(1, 2 + n_cand_cols, figsize=(fig_width, fig_height),
                              gridspec_kw={"width_ratios": width_ratios})
    ax_map = axes[0]
    ax_cands = axes[1:1 + n_cand_cols]
    ax_trunc = axes[-1]

    # --- Map panel ---
    ax_map.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=35, zorder=1)
    if nearest:
        nc = coords[nearest]
        ax_map.scatter(nc[:, 0], nc[:, 1], c="#3b6fa0", s=130, marker="o", zorder=3,
                        edgecolors="white", linewidths=0.8, label="Nearest candidate")
    if explore:
        ec = coords[explore]
        ax_map.scatter(ec[:, 0], ec[:, 1], c="#e07b1a", s=130, marker="^", zorder=3,
                        edgecolors="white", linewidths=0.8, label="Exploration candidate")

    full_path_coords = coords[full_nodes]
    _plot_directional_route(ax_map, full_path_coords, "#5b2d8e", linewidth=1.6, zorder=4)
    ax_map.scatter(*coords[curr_node], c="#1a1a1a", s=210, marker="*", zorder=6,
                    edgecolors="white", linewidths=1.0, label=f"Start node {curr_node}")

    label_nodes = [curr_node] + nearest + explore
    if len(label_nodes) > 40:  # keep the map legible on a real route with many candidates
        label_nodes = label_nodes[:40]
    for idx in label_nodes:
        ax_map.annotate(str(idx), (coords[idx, 0], coords[idx, 1]), fontsize=7.5,
                         color=COLOR_TEXT, xytext=(3, 3), textcoords="offset points", zorder=7)

    ax_map.set_title(f"QAOA sub-tour from node {curr_node}  (iteration {iteration}"
                      f"{', dual-filtered' if iteration >= 2 else ''})", fontsize=10.5)
    ax_map.legend(loc="best", fontsize=8)
    _style_axes(ax_map)

    # --- Candidate-selection economics, split into bounded-height column-groups ---
    cand_labels = ["Node", "Dist", "Dual", "D\u2212Dual", "Sel."]

    def _cand_formatter(row):
        node, dist, dual, net, role = row
        text = [str(node), f"{dist:.1f}", f"{dual:.1f}", f"{net:+.1f}", role]
        if role == "nearest":
            color = "#dbe8f5"
        elif role == "explore":
            color = "#fbe6d2"
        else:
            color = "#f2f2f2"
        return text, color

    chunks = [candidate_rows[i:i + rows_per_col] for i in range(0, len(candidate_rows), rows_per_col)] \
        if candidate_rows else [[]]
    for i, ax_c in enumerate(ax_cands):
        chunk_title = None
        if i == 0:
            chunk_title = (f"Candidate selection at node {curr_node}\n"
                            + ("\"Nearest\" = smallest (dist \u2212 dual), ranked -- not a sign filter"
                               if iteration >= 2 else
                               "Iteration 1: plain closest-by-distance (duals not yet used)"))
        _draw_minitable(ax_c, chunks[i] if i < len(chunks) else [], cand_labels, _cand_formatter, chunk_title,
                         col_widths=[0.18, 0.20, 0.20, 0.22, 0.20])

    if not candidate_rows:
        ax_cands[0].text(0.5, 0.5, "No other nodes available", transform=ax_cands[0].transAxes,
                          ha="center", va="center", fontsize=9, color="#888888", style="italic")

    # --- Prefix truncations (column pricing outcome) ---
    trunc_labels = ["Column", "Cost", "\u03a3\u03c0", "R.cost", "Kept?"]

    def _trunc_formatter(rec):
        path_str = "\u2192".join(str(x) for x in rec["nodes"])
        text = [path_str, f"{rec['cost']:.1f}", f"{rec['dual_sum']:.1f}",
                f"{rec['reduced_cost']:+.1f}", "kept" if rec["kept"] else "dropped"]
        color = "#e6f4e6" if rec["kept"] else "#f2f2f2"
        return text, color

    _draw_minitable(ax_trunc, truncation_records, trunc_labels, _trunc_formatter,
                     title=f"Prefix truncations (full path)\niteration {iteration} duals",
                     col_widths=[0.44, 0.14, 0.14, 0.16, 0.12])
    if not truncation_records:
        ax_trunc.text(0.5, 0.4, "No candidates qualified this iteration\n(every dist \u2212 dual \u2265 0)",
                       transform=ax_trunc.transAxes, ha="center", va="top", fontsize=9,
                       color="#a03030", style="italic")

    fig.suptitle(f"Step 2 \u2014 Pricing Subproblem: node {curr_node}, iteration {iteration}", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.86, wspace=0.35)
    _save_all_formats(
        fig, os.path.join(output_dir, f"02_{frame_idx:03d}_iter{iteration}_node{curr_node}"), formats
    )
    plt.close(fig)


# =====================================================================
# Pool growth across all iterations
# =====================================================================

def _frame_pool_growth(iteration_log, output_dir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    iters = [row["iteration"] for row in iteration_log]
    pool_sizes = [row["pool_size"] for row in iteration_log]
    new_cols = [row["num_new_columns"] for row in iteration_log]

    ax = axes[0]
    ax.plot(iters, pool_sizes, "-o", color="#3b6fa0", linewidth=2.2, markersize=6, zorder=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative pool size")
    ax.set_title("Column pool growth", fontsize=10.5)
    ax.set_xticks(iters)
    ax.grid(True, linestyle="--", linewidth=0.6, color=COLOR_GRID, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    colors = ["#1a7a1a" if c > 0 else "#b0b0b0" for c in new_cols]
    ax.bar(iters, new_cols, color=colors, width=0.6, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("New columns added")
    converged_at = next((row["iteration"] for row in iteration_log if row["num_new_columns"] == 0), None)
    title = "New columns per iteration"
    if converged_at is not None:
        title += f"  (converged at iteration {converged_at})"
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks(iters)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=COLOR_GRID, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Step 3 \u2014 Column Pool Growth Across Iterations", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_all_formats(fig, os.path.join(output_dir, "03_pool_growth"), formats)
    plt.close(fig)


# =====================================================================
# Dual evolution across all iterations, for a sample of nodes
# =====================================================================

def _frame_dual_evolution(dual_history, sample_nodes, depot_idx, output_dir, formats):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    iters = list(range(1, len(dual_history) + 1))

    for i, node in enumerate(sample_nodes):
        values = [dh.get(node, 0.0) for dh in dual_history]
        color = COLOR_DEPOT if node == depot_idx else ITERATION_CMAP(i / max(len(sample_nodes) - 1, 1))
        lw = 2.6 if node == depot_idx else 1.8
        ax.plot(iters, values, "-o", color=color, linewidth=lw, markersize=4.5,
                label=f"node {node}" + (" (depot)" if node == depot_idx else ""), zorder=3)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Dual price  $\\pi_i$")
    ax.set_title("Sampled nodes' dual prices across iterations", fontsize=11)
    ax.set_xticks(iters)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", linewidth=0.6, color=COLOR_GRID, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Step 4 \u2014 Dual Price Evolution", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_all_formats(fig, os.path.join(output_dir, "04_dual_evolution"), formats)
    plt.close(fig)


# =====================================================================
# Reduced-cost histogram across every priced column, all iterations
# =====================================================================

def _frame_reduced_cost_histogram(all_truncations, output_dir, formats):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    reduced_costs = [c["reduced_cost"] for c in all_truncations if len(c["nodes"]) > 1]
    if not reduced_costs:
        reduced_costs = [0.0]
    n_kept = sum(1 for c in reduced_costs if c < 0)
    n_dropped = len(reduced_costs) - n_kept

    ax.hist(reduced_costs, bins=30, color="#3b6fa0", edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color=COLOR_DEPOT, linestyle="--", linewidth=1.6, zorder=4,
               label="Reduced cost = 0 (improving threshold)")
    ax.set_xlabel("Reduced cost  (cost \u2212 \u03a3 duals)")
    ax.set_ylabel("Number of priced columns")
    ax.set_title(
        f"Pricing yield across all iterations \u2022 {n_kept} improving  /  {n_dropped} non-improving",
        fontsize=11,
    )
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=COLOR_GRID, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Step 5 \u2014 Reduced-Cost Distribution (All Iterations)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_all_formats(fig, os.path.join(output_dir, "05_reduced_cost_histogram"), formats)
    plt.close(fig)


# =====================================================================
# Final master ILP solution -- selected segments
# =====================================================================

def _frame_final_master(coords, depot_idx, selected_columns, n_pool, output_dir, formats):
    fig, ax = plt.subplots(figsize=(9.5, 8))
    for i, col in enumerate(selected_columns):
        nodes = col["nodes"]
        is_depot_seg = nodes[0] == depot_idx
        color = COLOR_DEPOT if is_depot_seg else SEGMENT_CMAP(i % 20)
        path_coords = coords[nodes]
        lw = 3.0 if is_depot_seg else 2.0
        ax.plot(path_coords[:, 0], path_coords[:, 1], "-o", color=color, linewidth=lw,
                markersize=5, zorder=3, alpha=0.95)
        ax.annotate(str(nodes[0]), (path_coords[0, 0], path_coords[0, 1]), fontsize=7.5,
                    color=COLOR_TEXT, xytext=(4, 4), textcoords="offset points", zorder=5)

    ax.scatter(*coords[depot_idx], c=COLOR_DEPOT, s=210, marker="D", zorder=6,
               edgecolors="white", linewidths=1.3)
    legend_handles = [
        Line2D([0], [0], color=COLOR_DEPOT, marker="D", markersize=9, linewidth=3, label="Depot segment"),
        Line2D([0], [0], color="#3b6fa0", marker="o", markersize=6, linewidth=2, label="Other selected segment"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.set_title(
        f"{len(selected_columns)} segments selected out of {n_pool} pool columns", fontsize=11,
    )
    _style_axes(ax)

    fig.suptitle("Step 6 \u2014 Final Master ILP Solution", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.90)
    _save_all_formats(fig, os.path.join(output_dir, "06_final_master_segments"), formats)
    plt.close(fig)


# =====================================================================
# Concatenation -- QAOA edges vs. greedy stitch edges
# =====================================================================

def _frame_concatenation(coords, depot_idx, selected_columns, raw_tour, matrix, output_dir, formats):
    depot_segments = [c for c in selected_columns if c["nodes"][0] == depot_idx]
    other_segments = [c for c in selected_columns if c["nodes"][0] != depot_idx]
    depot_segments.sort(key=lambda c: c["cost"])
    ordered_segments = [depot_segments[0]]
    remaining = list(other_segments) + depot_segments[1:]
    last_node = depot_segments[0]["nodes"][-1]
    while remaining:
        remaining.sort(key=lambda c: matrix[last_node, c["nodes"][0]])
        nxt = remaining.pop(0)
        ordered_segments.append(nxt)
        last_node = nxt["nodes"][-1]

    fig, ax = plt.subplots(figsize=(9.5, 8))
    junction_points = []
    for seg_idx, seg in enumerate(ordered_segments):
        nodes = seg["nodes"]
        path_coords = coords[nodes]
        ax.plot(path_coords[:, 0], path_coords[:, 1], "-", color="#1a6b1a", linewidth=2.4, zorder=3)
        ax.scatter(path_coords[:, 0], path_coords[:, 1], c="#1a6b1a", s=28, zorder=4)
        mid = path_coords[len(path_coords) // 2]
        ax.annotate(f"seg {seg_idx+1}", mid, fontsize=8, color="#1a6b1a", weight="bold",
                    xytext=(4, 6), textcoords="offset points", zorder=6)
        if seg_idx > 0:
            junction_points.append((ordered_segments[seg_idx - 1]["nodes"][-1], nodes[0]))

    for (a, b) in junction_points:
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                "--", color=COLOR_STITCH, linewidth=2.0, zorder=5)

    ax.scatter(*coords[depot_idx], c=COLOR_DEPOT, s=210, marker="D", zorder=7,
               edgecolors="white", linewidths=1.3)

    legend_handles = [
        Line2D([0], [0], color="#1a6b1a", linewidth=2.4, label="Within-segment edge (QAOA-found)"),
        Line2D([0], [0], color=COLOR_STITCH, linewidth=2.0, linestyle="--", label="Stitch edge (greedy concatenation)"),
        Line2D([0], [0], color=COLOR_DEPOT, marker="D", markersize=9, linewidth=0, label="Depot"),
    ]
    ax.legend(handles=legend_handles, loc="best")

    n_stitch = len(junction_points)
    n_within = len(raw_tour) - 1 - n_stitch
    ax.set_title(
        f"{len(ordered_segments)} segments chained  \u2022  {n_within} QAOA edges, {n_stitch} stitch edges",
        fontsize=11,
    )
    _style_axes(ax)

    fig.suptitle("Step 7 \u2014 Segment Concatenation Order", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.90)
    _save_all_formats(fig, os.path.join(output_dir, "07_concatenation"), formats)
    plt.close(fig)


# =====================================================================
# Before/after 2-opt
# =====================================================================

def _frame_two_opt_before_after(coords, depot_idx, raw_tour, final_tour, raw_cost, final_cost, output_dir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    for ax, tour, cost, title, color in [
        (axes[0], raw_tour, raw_cost, "Before 2-opt (segments concatenated)", "#c0621a"),
        (axes[1], final_tour, final_cost, "After 2-opt", "#1a6b1a"),
    ]:
        path_coords = coords[tour]
        _plot_directional_route(ax, path_coords, color, linewidth=1.6, zorder=3)
        ax.scatter(*coords[depot_idx], c=COLOR_DEPOT, s=190, marker="D", zorder=6,
                   edgecolors="white", linewidths=1.2)
        ax.set_title(f"{title}\nCost: {cost:.2f}", fontsize=10.5)
        _style_axes(ax)

    delta_pct = 100.0 * (raw_cost - final_cost) / raw_cost if raw_cost else 0.0
    fig.suptitle(f"Step 8 \u2014 Closing 2-opt Pass  \u2022  {delta_pct:.1f}% shorter", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.84)
    _save_all_formats(fig, os.path.join(output_dir, "08_two_opt_before_after"), formats)
    plt.close(fig)


# =====================================================================
# Main driver
# =====================================================================

def visualize_cg_stepwise_execution(
    data,
    qubit_count=4,
    exploration_percent=0.0,
    xy_mixer=False,
    only_improving_columns=True,
    n_iterations=ITERATION_CG,
    detail_iterations=None,
    max_detail_nodes=None,
    seed=101,
    output_dir="cg_visualizations",
    formats=("png",),
):
    """
    Runs the CG algorithm's logic step by step across all `n_iterations`
    pricing rounds (LP relaxation -> price over every node -> grow pool),
    then the final ILP round, concatenation, and 2-opt -- saving
    diagnostic frames along the way.

    By default, EVERY point is a rendered starting point at EVERY
    iteration -- no sampling of either dimension. For n points and
    n_iterations iterations, that is n * n_iterations pricing-detail
    frames (e.g. 10 points x 3 iterations = 30 frames, 100 points x 10
    iterations = 1000 frames). This matches the algorithm's actual
    structure exactly -- every point IS a starting point at every
    iteration in cg_hybrid_lrwsqaoa_sub.py's _generate_priced_columns,
    with no depletion of the candidate pool across different starting
    points -- and the visualization now shows all of it rather than a
    sample.

    `detail_iterations` and `max_detail_nodes` remain available if you
    explicitly want to restrict rendering (e.g. for a very large route
    where thousands of frames aren't practical) -- pass an explicit set
    of iteration numbers, and/or an integer to cap how many points get a
    frame per rendered iteration. Left as None (the default), both are
    unrestricted: every iteration, every point.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    matrix = data["matrix"]
    coords = data["coords"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)
    n_iterations = max(1, n_iterations)
    global_max_dist = float(matrix.max()) if n > 1 else 0.0

    if detail_iterations is None:
        detail_iterations = set(range(1, n_iterations + 1))  # every iteration, no sampling
    else:
        detail_iterations = set(detail_iterations)

    if max_detail_nodes is None:
        detail_nodes = set(range(n))  # every point, no sampling -- used for pricing detail frames
    else:
        detail_nodes = set(np.linspace(0, n - 1, num=min(max_detail_nodes, n), dtype=int).tolist())
        detail_nodes.add(depot_idx)

    # Deliberately SEPARATE from detail_nodes above. detail_nodes controls the
    # per-node pricing frames and must stay exhaustive (every point, every
    # iteration) -- that's a hard requirement, not a display convenience.
    # This one is only for the dual-evolution SUMMARY chart (a trend line
    # per node on one shared plot), where plotting all n nodes stops being
    # readable well before n gets anywhere near real Amazon-route sizes: at
    # n=40 the legend alone already swallows the chart. Capped at a small,
    # evenly-spaced sample + depot regardless of how many points detail_nodes
    # covers, since this chart's job is showing a representative trend, not
    # an exhaustive per-node account (the pricing frames already are that).
    dual_evolution_sample = set(np.linspace(0, n - 1, num=min(10, n), dtype=int).tolist())
    dual_evolution_sample.add(depot_idx)

    t_start = time.time()
    pool = cg._build_initial_columns(n, matrix, depot_idx)
    dual_history = []
    iteration_log = []
    all_truncations_for_stats = []
    pricing_frame_counter = 0

    print(f"Running {n_iterations} pricing iteration(s) (detail frames at iterations "
          f"{sorted(detail_iterations)}, nodes {sorted(detail_nodes)})...")

    for it in range(1, n_iterations + 1):
        status_lp, _, _, duals = cg._solve_master(pool, list(range(n)), relaxation=True)
        dual_history.append(dict(duals))
        render_detail_this_iter = it in detail_iterations

        if render_detail_this_iter:
            _frame_duals_snapshot(depot_idx, duals, n, it, len(pool), output_dir, formats)

        priced_this_iter = []
        apply_dual_candidate_filter = (it >= 2)

        # QAOA itself incorporates duals from iteration >= 2 onward, mirroring
        # cg_hybrid_lrwsqaoa_sub.py's _generate_priced_columns exactly: solve
        # over a dual-adjusted matrix (column j reduced by duals[j]) instead
        # of raw distances, so QAOA searches for low REDUCED-cost orderings.
        # Cost bookkeeping below still always uses the raw `matrix`.
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

            # Uses the SAME dual-aware selection as cg_hybrid_lrwsqaoa_sub.py's
            # _generate_priced_columns (imported directly, not reimplemented),
            # so the pool this driver builds up is byte-for-byte what
            # run_cg_hybrid_lrwsqaoa_sub() would produce with the same seed.
            # Every curr_node searches its own full candidate pool
            # independently here -- nothing is removed from later nodes'
            # searches because an earlier node happened to select it.
            nearest, explore = cg._dual_aware_nearest_and_explore(
                curr_node, exclude, matrix, k_batch, exploration_percent, rng,
                duals=(duals if apply_dual_candidate_filter else None),
                global_max_dist=global_max_dist,
            )
            candidates = nearest + explore

            want_detail = render_detail_this_iter and curr_node in detail_nodes
            if want_detail:
                # Candidate-selection economics for this node: distance,
                # dual, and distance-minus-dual for EVERY other node under
                # consideration -- all of them, not a sampled subset. This
                # is what makes the "smallest reduced cost among all points"
                # ranking (used from iteration >= 2) fully visible, matching
                # what the algorithm itself actually searches over (every
                # other node in the graph, subject only to RADIUS_POOL_SEARCH,
                # which defaults to no restriction).
                #
                # Computed BEFORE the "no candidates" check below and
                # rendered regardless of whether anything qualified.
                others = [i for i in range(n) if i != curr_node and i not in exclude]
                others_sorted = sorted(others, key=lambda x: (matrix[curr_node, x], x))
                selected_set = set(nearest) | set(explore)
                if apply_dual_candidate_filter:
                    # Rank order (smallest dist-dual first), same as the
                    # actual selection criterion -- not sorted by distance.
                    ordered_for_table = sorted(
                        others_sorted,
                        key=lambda x: (matrix[curr_node, x] - duals.get(x, 0.0), matrix[curr_node, x], x),
                    )
                else:
                    ordered_for_table = others_sorted
                candidate_rows = [
                    (x, float(matrix[curr_node, x]), float(duals.get(x, 0.0)),
                     float(matrix[curr_node, x]) - float(duals.get(x, 0.0)),
                     "nearest" if x in nearest else ("explore" if x in explore else "-"))
                    for x in ordered_for_table
                ]

            if not candidates:
                if want_detail:
                    pricing_frame_counter += 1
                    _frame_pricing_node(pricing_frame_counter, coords, matrix, curr_node, nearest, explore,
                                         [curr_node], [], candidate_rows, it, output_dir, formats)
                continue

            subtour = solve_wslr_qaoa_subtour(curr_node, candidates, qaoa_matrix, xy_mixer=xy_mixer)
            if not subtour:
                if want_detail:
                    pricing_frame_counter += 1
                    _frame_pricing_node(pricing_frame_counter, coords, matrix, curr_node, nearest, explore,
                                         [curr_node], [], candidate_rows, it, output_dir, formats)
                continue
            full_nodes = [curr_node] + subtour

            truncation_records = []
            for L in range(len(full_nodes), 0, -1):
                seg = full_nodes[:L]
                seg_cost = cg._open_path_cost(seg, matrix)  # raw matrix -- real cost
                dual_sum = sum(duals.get(node, 0.0) for node in seg)
                reduced_cost = seg_cost - dual_sum
                kept = (reduced_cost < -1e-9) or (L == 1)
                record = {"nodes": seg, "cost": seg_cost, "dual_sum": dual_sum,
                          "reduced_cost": reduced_cost, "kept": kept, "start": curr_node}
                truncation_records.append(record)
                all_truncations_for_stats.append(record)
                if (not only_improving_columns) or kept:
                    priced_this_iter.append(record)

            if want_detail:
                pricing_frame_counter += 1
                _frame_pricing_node(pricing_frame_counter, coords, matrix, curr_node, nearest, explore,
                                     full_nodes, truncation_records, candidate_rows, it, output_dir, formats)

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


if __name__ == "__main__":
    data = generate_random_tsp_data(n_nodes=16, seed=101)
    visualize_cg_stepwise_execution(
        data, qubit_count=4, exploration_percent=0.2, xy_mixer=False,
        only_improving_columns=True, n_iterations=ITERATION_CG,
        seed=101,
        # max_detail_nodes and detail_iterations intentionally NOT set here --
        # leaving them at their function defaults (None) means every point at
        # every iteration gets rendered, exactly as required. A previous
        # version of this __main__ block hardcoded max_detail_nodes=6, which
        # silently overrode the fixed default whenever this script was run
        # directly (`python visualize_step_by_step_CG.py`) rather than
        # imported and called from elsewhere -- that stale hardcoded value,
        # not the function's actual default, is what was producing the
        # sampled 0, 3, 6, 9... pattern.
    )
