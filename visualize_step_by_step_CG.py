"""
visualize_step_by_step_CG.py

Step-by-step diagnostic visualizer for the Column-Generation Hybrid
algorithm (cg_hybrid_lrwsqaoa_sub.py). Unlike a route-comparison plot,
this is meant to show the *mechanics* of the algorithm as it runs:

  1. Round-1 master LP duals -- what price the master initially assigns
     to covering each node.
  2. The pricing subproblem in detail, for a sample of starting nodes --
     which candidates were considered, what path QAOA found, and for
     every truncation of that path, its cost, the sum of duals it
     covers, its reduced cost, and whether it survived the filter.
  3. A reduced-cost histogram across every priced column (not just the
     detailed sample), so you can see the overall pricing yield.
  4. The Round-2 master ILP's selected segments, each drawn in its own
     color.
  5. How those segments get concatenated into one tour -- QAOA-found
     edges (within a segment) are drawn differently from the greedy
     "stitch" edges added between segments, so you can see how much of
     the final tour is quantum-derived versus glued together.
  6. Before/after cost effect of the closing 2-opt pass.

Does not modify cg_hybrid_lrwsqaoa_sub.py or algo_hybrid_LRWSQAOA.py --
it imports and drives their existing functions directly (including a
few underscore-prefixed internals from cg_hybrid_lrwsqaoa_sub.py, since
this is a companion diagnostic script for that exact module, the same
way visualize_step_by_step.py drives solve_wslr_qaoa_subtour from
algo_hybrid_LRWSQAOA.py without modifying it).

Saves frames into 'cg_visualizations/' without GUI blocking.
"""

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
from plot_publication import (
    _style_axes,
    _plot_directional_route,
    _save_all_formats,
    COLOR_DEPOT,
    COLOR_GRID,
    COLOR_TEXT,
)

SEGMENT_CMAP = plt.get_cmap("tab20")
COLOR_KEPT = "#1a7a1a"
COLOR_DROPPED = "#b0b0b0"
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
# Frame 1: Round-1 duals
# =====================================================================

def _frame_initial_duals(coords, depot_idx, duals, n, output_dir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    ax = axes[0]
    for i in range(n):
        if i == depot_idx:
            continue
        ax.plot([coords[depot_idx, 0], coords[i, 0]], [coords[depot_idx, 1], coords[i, 1]],
                color="#9db8d8", linewidth=0.9, alpha=0.7, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], c="#3b6fa0", s=55, zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(*coords[depot_idx], c=COLOR_DEPOT, s=200, marker="D", zorder=5,
               edgecolors="white", linewidths=1.3, label="Depot")
    ax.set_title("Initial Columns (Singletons, Cost = Distance from Depot)", fontsize=10.5)
    ax.legend(loc="best")
    _style_axes(ax)

    ax = axes[1]
    node_ids = list(range(n))
    values = [duals.get(i, 0.0) for i in node_ids]
    colors = [COLOR_DEPOT if i == depot_idx else "#3b6fa0" for i in node_ids]
    ax.bar(node_ids, values, color=colors, width=0.75, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Node index")
    ax.set_ylabel("Dual price  $\\pi_i$")
    ax.set_title("Round-1 LP Duals  ($\\pi_i$ = distance(depot, i))", fontsize=10.5)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=COLOR_GRID, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Step 1 — Round 1: Master LP Relaxation & Duals", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.85)
    _save_all_formats(fig, os.path.join(output_dir, "01_round1_duals"), formats)
    plt.close(fig)


# =====================================================================
# Frame 2 (repeated): pricing subproblem detail for a sampled node
# =====================================================================

def _frame_pricing_node(
    frame_idx, coords, matrix, curr_node, nearest, explore, full_nodes,
    truncation_records, output_dir, formats,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # --- Left: map of this pricing subproblem ---
    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=35, zorder=1)
    if nearest:
        nc = coords[nearest]
        ax.scatter(nc[:, 0], nc[:, 1], c="#3b6fa0", s=130, marker="o", zorder=3,
                   edgecolors="white", linewidths=0.8, label="Nearest candidate")
    if explore:
        ec = coords[explore]
        ax.scatter(ec[:, 0], ec[:, 1], c="#e07b1a", s=130, marker="^", zorder=3,
                   edgecolors="white", linewidths=0.8, label="Exploration candidate")

    full_path_coords = coords[full_nodes]
    _plot_directional_route(ax, full_path_coords, cmap_name="Purples", linewidth=2.4, zorder=4)
    ax.scatter(*coords[curr_node], c="#1a1a1a", s=210, marker="*", zorder=6,
               edgecolors="white", linewidths=1.0, label=f"Start node {curr_node}")

    for idx in [curr_node] + nearest + explore:
        ax.annotate(str(idx), (coords[idx, 0], coords[idx, 1]), fontsize=7.5,
                    color=COLOR_TEXT, xytext=(3, 3), textcoords="offset points", zorder=7)

    ax.set_title(f"QAOA sub-tour from node {curr_node}", fontsize=10.5)
    ax.legend(loc="best", fontsize=8)
    _style_axes(ax)

    # --- Right: truncation table ---
    ax2 = axes[1]
    ax2.axis("off")
    col_labels = ["Column", "Cost", "\u03a3 duals", "Red. cost", "Kept?"]
    cell_text = []
    cell_colors = []
    for rec in truncation_records:
        path_str = "\u2192".join(str(x) for x in rec["nodes"])
        cell_text.append([
            path_str,
            f"{rec['cost']:.1f}",
            f"{rec['dual_sum']:.1f}",
            f"{rec['reduced_cost']:+.1f}",
            "kept" if rec["kept"] else "dropped",
        ])
        row_color = "#e6f4e6" if rec["kept"] else "#f2f2f2"
        cell_colors.append([row_color] * len(col_labels))

    table = ax2.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                       cellLoc="center", loc="center",
                       colWidths=[0.34, 0.14, 0.16, 0.20, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2B2B2B")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        cell.set_edgecolor("#cccccc")

    ax2.set_title("Prefix truncations priced against Round-1 duals", fontsize=10.5, pad=14)
    ax2.set_xlim(0, 1)

    fig.suptitle(f"Step 2.{frame_idx} — Pricing Subproblem: start node {curr_node}", fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.86, wspace=0.25)
    _save_all_formats(fig, os.path.join(output_dir, f"02_{frame_idx:02d}_pricing_node{curr_node}"), formats)
    plt.close(fig)


# =====================================================================
# Frame 3: reduced-cost histogram across all priced columns
# =====================================================================

def _frame_reduced_cost_histogram(all_priced, output_dir, formats):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    reduced_costs = [c["reduced_cost"] for c in all_priced if len(c["nodes"]) > 1]
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
        f"Pricing yield across all nodes  \u2022  {n_kept} improving  /  {n_dropped} non-improving",
        fontsize=11,
    )
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=COLOR_GRID, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Step 3 — Reduced-Cost Distribution (Pricing Summary)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_all_formats(fig, os.path.join(output_dir, "03_reduced_cost_histogram"), formats)
    plt.close(fig)


# =====================================================================
# Frame 4: final master ILP solution -- selected segments
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
        f"{len(selected_columns)} segments selected out of {n_pool} priced columns", fontsize=11,
    )
    _style_axes(ax)

    fig.suptitle("Step 4 — Round 2: Final Master ILP Solution", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.90)
    _save_all_formats(fig, os.path.join(output_dir, "04_final_master_segments"), formats)
    plt.close(fig)


# =====================================================================
# Frame 5: concatenation -- QAOA edges vs. greedy stitch edges
# =====================================================================

def _frame_concatenation(coords, depot_idx, selected_columns, raw_tour, matrix, output_dir, formats):
    # Recompute concatenation order the same way _concatenate_segments does,
    # but also record which edges are "stitch" edges (segment boundaries)
    # vs. "within-segment" (QAOA-found) edges, for visualization.
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

    fig.suptitle("Step 5 — Segment Concatenation Order", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.90)
    _save_all_formats(fig, os.path.join(output_dir, "05_concatenation"), formats)
    plt.close(fig)


# =====================================================================
# Frame 6: before/after 2-opt
# =====================================================================

def _frame_two_opt_before_after(coords, depot_idx, raw_tour, final_tour, raw_cost, final_cost, output_dir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    for ax, tour, cost, title, cmap in [
        (axes[0], raw_tour, raw_cost, "Before 2-opt (segments concatenated)", "Oranges"),
        (axes[1], final_tour, final_cost, "After 2-opt", "Greens"),
    ]:
        path_coords = coords[tour]
        _plot_directional_route(ax, path_coords, cmap_name=cmap, linewidth=2.4, zorder=3)
        ax.scatter(*coords[depot_idx], c=COLOR_DEPOT, s=190, marker="D", zorder=6,
                   edgecolors="white", linewidths=1.2)
        ax.set_title(f"{title}\nCost: {cost:.2f}", fontsize=10.5)
        _style_axes(ax)

    delta_pct = 100.0 * (raw_cost - final_cost) / raw_cost if raw_cost else 0.0
    fig.suptitle(f"Step 6 — Closing 2-opt Pass  \u2022  {delta_pct:.1f}% shorter", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.84)
    _save_all_formats(fig, os.path.join(output_dir, "06_two_opt_before_after"), formats)
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
    max_detail_nodes=6,
    seed=101,
    output_dir="cg_visualizations",
    formats=("png",),
):
    """
    Runs the CG algorithm's logic step by step (Round 1 -> pricing over
    every node -> Round 2 -> concatenation -> 2-opt), saving a diagnostic
    frame at each stage. Detailed per-node pricing frames are rendered
    for an evenly-spaced sample of `max_detail_nodes` starting nodes
    (always including the depot), but pricing itself still runs over
    EVERY node so the resulting column pool and final tour are identical
    to what run_cg_hybrid_lrwsqaoa_sub() would produce with the same
    seed and parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    matrix = data["matrix"]
    coords = data["coords"]
    n = data["n_nodes"]
    depot_idx = data.get("depot_idx", 0)

    t_start = time.time()

    # --- Round 1 ---
    print("Step 1: Round-1 LP relaxation for duals...")
    initial_columns = cg._build_initial_columns(n, matrix, depot_idx)
    status1, _, _, duals = cg._solve_master(initial_columns, list(range(n)), relaxation=True)
    print(f"  status={status1}, sample duals={[round(duals.get(i,0),1) for i in range(min(n,5))]}")
    _frame_initial_duals(coords, depot_idx, duals, n, output_dir, formats)

    # --- Pricing over every node, with detail frames for a sample ---
    detail_nodes = set(np.linspace(0, n - 1, num=min(max_detail_nodes, n), dtype=int).tolist())
    detail_nodes.add(depot_idx)

    all_priced = []
    all_truncations_for_stats = []
    frame_idx = 0
    print(f"Step 2: Pricing subproblem over all {n} nodes "
          f"(detailed frames for {sorted(detail_nodes)})...")
    for curr_node in range(n):
        exclude = {depot_idx} if curr_node != depot_idx else set()
        k_batch = min(qubit_count, n - 1 - len(exclude))
        if k_batch <= 0:
            continue

        others = [i for i in range(n) if i != curr_node and i not in exclude]
        others_sorted = sorted(others, key=lambda x: (matrix[curr_node, x], x))
        if exploration_percent <= 0.0 or k_batch <= 1:
            nearest, explore = others_sorted[:k_batch], []
        else:
            import math
            n_explore = min(int(math.floor(k_batch * exploration_percent)), k_batch - 1)
            n_nearest = k_batch - n_explore
            nearest = others_sorted[:n_nearest]
            remaining = others_sorted[n_nearest:]
            if n_explore > 0 and remaining:
                idx = rng.choice(len(remaining), size=min(n_explore, len(remaining)), replace=False)
                explore = [remaining[i] for i in idx]
            else:
                explore = []
        candidates = nearest + explore
        if not candidates:
            continue

        subtour = solve_wslr_qaoa_subtour(curr_node, candidates, matrix, xy_mixer=xy_mixer)
        if not subtour:
            continue
        full_nodes = [curr_node] + subtour

        truncation_records = []
        for L in range(len(full_nodes), 0, -1):
            seg = full_nodes[:L]
            seg_cost = cg._open_path_cost(seg, matrix)
            dual_sum = sum(duals.get(node, 0.0) for node in seg)
            reduced_cost = seg_cost - dual_sum
            kept = (reduced_cost < -1e-9) or (L == 1)
            record = {"nodes": seg, "cost": seg_cost, "dual_sum": dual_sum,
                      "reduced_cost": reduced_cost, "kept": kept, "start": curr_node}
            truncation_records.append(record)
            all_truncations_for_stats.append(record)
            if (not only_improving_columns) or kept:
                all_priced.append(record)

        if curr_node in detail_nodes:
            frame_idx += 1
            _frame_pricing_node(frame_idx, coords, matrix, curr_node, nearest, explore,
                                 full_nodes, truncation_records, output_dir, formats)
            print(f"  [detail frame {frame_idx}] node {curr_node}: "
                  f"{sum(r['kept'] for r in truncation_records)}/{len(truncation_records)} kept")

    _frame_reduced_cost_histogram(all_truncations_for_stats, output_dir, formats)

    full_pool = cg._dedupe_columns(initial_columns + all_priced)
    print(f"  pool size after dedupe: {len(full_pool)}")

    # --- Round 2 ---
    print("Step 4: Round-2 binary ILP master solve...")
    status2, selected_idx, _, _ = cg._solve_master(full_pool, list(range(n)), relaxation=False)
    if status2 == "Optimal" and selected_idx:
        selected_columns = [full_pool[i] for i in selected_idx]
    else:
        warnings.warn(f"Round-2 status '{status2}'; using greedy fallback for this visualization.")
        selected_columns = cg._greedy_set_cover_fallback(full_pool, list(range(n)))
    print(f"  status={status2}, segments selected={len(selected_columns)}")
    _frame_final_master(coords, depot_idx, selected_columns, len(full_pool), output_dir, formats)

    # --- Concatenation ---
    print("Step 5: Concatenating segments...")
    raw_tour = cg._concatenate_segments(selected_columns, depot_idx, matrix)
    raw_cost = cg._open_path_cost(raw_tour, matrix)
    _frame_concatenation(coords, depot_idx, selected_columns, raw_tour, matrix, output_dir, formats)

    # --- 2-opt ---
    print("Step 6: 2-opt polish...")
    final_tour = cg._two_opt_open_tsp(raw_tour, matrix)
    final_cost = cg._open_path_cost(final_tour, matrix)
    _frame_two_opt_before_after(coords, depot_idx, raw_tour, final_tour, raw_cost, final_cost, output_dir, formats)

    print(f"\nDone in {time.time()-t_start:.2f}s. Raw cost {raw_cost:.2f} -> Final cost {final_cost:.2f} "
          f"({100*(raw_cost-final_cost)/raw_cost:.1f}% from 2-opt). Frames saved under '{output_dir}/'")

    return {
        "final_tour": final_tour,
        "final_cost": final_cost,
        "raw_tour": raw_tour,
        "raw_cost": raw_cost,
        "duals": duals,
        "num_pool_columns": len(full_pool),
        "num_segments_selected": len(selected_columns),
    }


if __name__ == "__main__":
    data = generate_random_tsp_data(n_nodes=16, seed=101)
    visualize_cg_stepwise_execution(
        data, qubit_count=4, exploration_percent=0.2, xy_mixer=False,
        only_improving_columns=True, max_detail_nodes=6, seed=101,
    )
