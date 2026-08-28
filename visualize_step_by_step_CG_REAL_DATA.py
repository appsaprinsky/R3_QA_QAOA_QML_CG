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

--------------------------------------------------------------------------
FIX LOG (this revision)
--------------------------------------------------------------------------
* THE ACTUAL ROOT CAUSE of "cannot see a damn thing" on real routes was
  upstream of this file, in algo_data_loader.py: a node missing from
  stops metadata kept its default coordinate of (0, 0), which -- when
  only SOME (not all) of a route's nodes were affected, a case the old
  `np.all(coords == 0)` fallback below never caught -- silently entered
  every plot's coordinate data and stretched auto-computed axis limits
  from "one delivery zone" to "includes the Gulf of Guinea". Fixed at
  the source in AmazonDataLoader.extract_single_route() (see that
  file's own FIX LOG); the `np.all(coords == 0)` check in
  load_one_real_route() below is now a redundant safety net, not the
  primary fix.
* _frame_pricing_map_only(): an EARLIER revision's zoom fix used a fixed
  absolute padding floor (`pad = max(span * 0.6, 3.0)`), calibrated
  against synthetic test coordinates in the range [0, 100]. Real
  coordinates can be in a completely different scale (raw lat/lng
  degrees, where a whole route might span only ~0.01-0.05 total, or
  MDS-embedded units of some other magnitude entirely) -- against data
  like that, an absolute floor of "3.0" is enormous, and would have
  quietly re-introduced the exact same "can't see the candidates"
  problem the zoom fix was meant to solve, just by a different
  mechanism (over-zooming OUT this time instead of never zooming in).
  Padding is now computed as a fraction of the FULL ROUTE's own
  coordinate span (passed in once per run as `full_coord_span`, not
  recomputed from `coords` on every one of the (up to hundreds of)
  pricing-frame calls in a run) instead of an absolute number, so it
  scales correctly regardless of what units `coords` happens to be in.
* _frame_pricing_map_only(): a real ordering bug in the PREVIOUS
  revision -- ax.set_aspect("equal", adjustable="box") + the explicit
  zoom's set_xlim/set_ylim were applied, and THEN _style_axes(ax) was
  called, which sets adjustable="datalim" internally and silently
  recomputed the limits back to a full data-fit, undoing the zoom fix
  entirely. Caught this by testing against the real imported
  _style_axes rather than a standalone re-implementation. _style_axes()
  is now called BEFORE the explicit zoom override, so the override is
  what actually sticks.
* plot_publication.py's _plot_directional_route(): arrow shaft length
  was a fixed fraction of the AXIS diagonal only, unrelated to how long
  the specific segment it was drawn on actually was -- on the tightly
  zoomed views these pricing frames now use, with only 2-4 short
  segments, that fixed length could exceed the segment itself, making
  arrows look oversized and dominate the short edges they sat on. Now
  also capped to a fraction of each segment's own length, with a
  smaller arrowhead. Fixes this for every figure that uses
  _plot_directional_route, not just this file.
* Path clarity: _frame_pricing_map_only() now labels each point on the
  QAOA sub-tour with its VISIT ORDER (not just node id), marks the
  segment's end distinctly from its start (not just the start, as
  before), and prints the resolved path as on-figure text so the
  reading order is unambiguous without having to trace arrows.
  _frame_final_master(), _frame_concatenation(), and
  _frame_two_opt_before_after() -- previously imported unmodified from
  visualize_step_by_step_CG.py -- are replaced with local
  _real_data versions (see below) that additionally PRINT and SAVE
  (as a companion .txt file next to the images) the exact node
  sequence for every selected master-problem segment and for the final
  concatenated/2-opt-polished route, plus order-position labels on the
  route maps themselves. This is what you'd actually open to "orient on
  the generated path" rather than only staring at the map.
--------------------------------------------------------------------------
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
from plot_publication import _style_axes, _plot_directional_route, _mark_start_end, _save_all_formats, COLOR_TEXT

# Reused UNMODIFIED from visualize_step_by_step_CG.py -- these are all
# already graphs (bar chart, line charts, histogram), no tables, so
# nothing about them needed to change.
from visualize_step_by_step_CG import (
    _frame_duals_snapshot,
    _frame_pool_growth,
    _frame_dual_evolution,
    _frame_reduced_cost_histogram,
)
# _frame_final_master, _frame_concatenation, _frame_two_opt_before_after
# are NOT imported from visualize_step_by_step_CG.py anymore -- see the
# locally-defined _real_data replacements below (FIX LOG: these now
# print/save the actual selected node paths, not just draw them, and use
# order-position labels so start/continue/end is unambiguous).

# --- CRITICAL CPU & THERMAL LIMITS (matches run_amazon_experiment.py / run_CG_experiment.py) ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


# =====================================================================
# Shared helper: print AND save a node path so it's easy to "orient on
# the generated path" without having to trace arrows on a map -- printed
# to stdout immediately, and appended to a companion .txt file next to
# the images (console output scrolls away during a long run; the file
# doesn't).
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
    """
    Labels points along `path_nodes` with their VISIT ORDER (0-indexed
    position in path_nodes) rather than raw node id -- "where am I in
    the sequence" is what actually answers "where does this start,
    where does it continue, where does it end" for a long real route,
    which a raw node-id label doesn't by itself. Always labels the
    first and last position regardless of subsampling; labels every
    point if there are few enough to stay readable, otherwise evenly
    subsamples like plot_publication.py's own _annotate_sparse.
    """
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
# Pricing frame: graphs only, no tables
# =====================================================================

def _frame_pricing_map_only(frame_idx, coords, matrix, curr_node, nearest, explore, full_nodes,
                             iteration, output_dir, formats, full_coord_span):
    """
    `full_coord_span` is the FULL ROUTE's own coordinate span (max over
    both axes of max-min), computed ONCE by the caller and passed in --
    see FIX LOG above for why padding is scaled against this rather than
    a fixed absolute number.

    Every candidate here is, by construction, part of `full_nodes` (the
    resolved QAOA sub-tour) -- there's no "candidate not on the path"
    case to distinguish, so every labeled point gets an unambiguous
    visit-order tag instead of just its raw node id (see FIX LOG).
    """
    fig, ax = plt.subplots(figsize=(9.5, 8.2))

    ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=30, zorder=1)

    full_path_coords = coords[full_nodes]
    _plot_directional_route(ax, full_path_coords, "#5b2d8e", linewidth=1.9, zorder=4)

    # Start (triangle) and end (square) -- same marker language as
    # plot_publication.py's route panels, via the shared _mark_start_end,
    # so "which end is which" reads the same way across every figure in
    # this pipeline.
    _mark_start_end(ax, full_path_coords, "#5b2d8e", zorder=6)
    ax.scatter(*coords[curr_node], c="#1a1a1a", s=90, marker="*", zorder=7,
               edgecolors="white", linewidths=0.9, label=f"Start node {curr_node}")

    if nearest:
        nc = coords[nearest]
        ax.scatter(nc[:, 0], nc[:, 1], c="#3b6fa0", s=30, marker="o", zorder=5, label="Nearest candidate")
    if explore:
        ec = coords[explore]
        ax.scatter(ec[:, 0], ec[:, 1], c="#e07b1a", s=30, marker="^", zorder=5, label="Exploration candidate")

    _annotate_order_sparse(ax, coords, full_nodes, max_labels=len(full_nodes))  # short path -- label every step

    ax.set_title(f"QAOA sub-tour from node {curr_node}"
                 f"{' (dual-filtered)' if iteration >= 2 else ''}", fontsize=12.5)
    ax.legend(loc="best", fontsize=9)

    # _style_axes() is called BEFORE the explicit zoom below, not after --
    # it sets ax.set_aspect("equal", adjustable="datalim") internally,
    # which (if called AFTER our explicit set_xlim/set_ylim/set_aspect)
    # silently recomputes and overrides them back to a data-fitted view,
    # undoing the zoom entirely. Caught this via a direct test against
    # the real _style_axes (not a standalone re-implementation) -- an
    # earlier version of this function had the two calls in the wrong
    # order and would have quietly reintroduced the exact "can't see the
    # candidates" problem the zoom fix exists to solve.
    _style_axes(ax)

    focus_idx = [curr_node] + nearest + explore
    focus_coords = coords[focus_idx]
    xmin, xmax = focus_coords[:, 0].min(), focus_coords[:, 0].max()
    ymin, ymax = focus_coords[:, 1].min(), focus_coords[:, 1].max()
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    # Padding is the larger of (a) 60% of the local candidate cluster's
    # own span, so a wide cluster still gets breathing room, and
    # (b) 3% of the FULL ROUTE's span, so a very tight local cluster
    # (e.g. three candidates a few meters apart on a 250-stop route)
    # still gets a sensible minimum window instead of zooming in so far
    # the background context disappears. Both terms scale with the
    # data's own units -- no fixed absolute number, unlike the previous
    # revision (see FIX LOG).
    pad = max(span * 0.6, full_coord_span * 0.03)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = span / 2 + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")  # "box", not "datalim" -- must come AFTER _style_axes above

    # On-figure path readout -- so the reading order is legible without
    # having to trace arrows at all.
    path_str = " \u2192 ".join(str(x) for x in full_nodes)
    ax.text(0.02, 0.02, f"Path: {path_str}", transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left", color=COLOR_TEXT,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))

    fig.suptitle(f"Step 2 \u2014 Pricing Subproblem: node {curr_node}, iteration {iteration}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_all_formats(
        fig, os.path.join(output_dir, f"02_{frame_idx:03d}_iter{iteration}_node{curr_node}"), formats
    )
    plt.close(fig)


# =====================================================================
# Final master / concatenation / 2-opt frames: local _real_data versions
# (see FIX LOG -- these now print/save the actual node paths, and use
# order-position labels, instead of only drawing a map).
# =====================================================================

_SEGMENT_COLORS = plt.get_cmap("tab20").colors  # 20 distinguishable colors, cycled if more segments


def _frame_final_master(coords, depot_idx, selected_columns, pool_size, output_dir, formats):
    """
    Local _real_data replacement for visualize_step_by_step_CG.py's
    version of this frame (that file itself is not modified -- see
    module docstring). Draws every SELECTED column/segment (the Master
    Problem's chosen columns, pre-concatenation) in its own color, and
    prints + saves every segment's exact node path to
    '<output_dir>/paths_final_master.txt' -- open that file (or read the
    console output) for the segment paths a legend/map alone can't show
    clearly once there are more than a handful of segments.
    """
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=22, zorder=1)

    txt_filename = "paths_final_master.txt"
    open(os.path.join(output_dir, txt_filename), "w").close()  # fresh file each run
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
        if i < 20:  # cap the on-figure legend; the text file has every segment regardless
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
    """
    Local _real_data replacement (see _frame_final_master's docstring
    for why). Draws the single concatenated tour (segments stitched
    together in visit order) as ONE continuous path with visit-order
    labels, and prints + saves the exact resulting node sequence to
    '<output_dir>/paths_route.txt'.
    """
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=22, zorder=1)

    path_coords = coords[raw_tour]
    _plot_directional_route(ax, path_coords, "#1f7a5c", linewidth=2.2, zorder=3)
    _mark_start_end(ax, path_coords, "#1f7a5c", zorder=6)
    _annotate_order_sparse(ax, coords, raw_tour, max_labels=30)

    ax.scatter(*coords[depot_idx], c="#C1272D", s=260, marker="D", edgecolors="white",
               linewidths=1.4, zorder=8, label="Depot")
    raw_cost = cg._open_path_cost(raw_tour, matrix)
    ax.set_title(f"Step 4 \u2014 Concatenated Route (pre-2-opt): {len(raw_tour)} stops, cost {raw_cost:.1f}",
                 fontsize=13)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f7a5c",
                   markeredgecolor="white", markersize=10, label="Start"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#1f7a5c",
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
    """
    Local _real_data replacement (see _frame_final_master's docstring
    for why). Side-by-side before/after 2-opt, both with visit-order
    labels; prints + saves the FINAL route -- the one actually
    delivered -- to '<output_dir>/paths_route.txt' alongside the
    pre-2-opt path already written by _frame_concatenation above, so
    that file ends up with both in one place.
    """
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5))

    for ax, tour, cost, title in [
        (axes[0], raw_tour, raw_cost, "Before 2-opt"),
        (axes[1], final_tour, final_cost, "After 2-opt"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=22, zorder=1)
        path_coords = coords[tour]
        _plot_directional_route(ax, path_coords, "#1f7a5c", linewidth=2.0, zorder=3)
        _mark_start_end(ax, path_coords, "#1f7a5c", zorder=6)
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

    # Computed ONCE per run, not per pricing-frame call -- see FIX LOG
    # in _frame_pricing_map_only for why this replaced an absolute
    # padding floor.
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
                                             nearest, explore, [curr_node], it, output_dir, formats,
                                             full_coord_span)
                continue

            subtour = solve_wslr_qaoa_subtour(curr_node, candidates, qaoa_matrix, xy_mixer=xy_mixer)
            if not subtour:
                if want_detail:
                    pricing_frame_counter += 1
                    _frame_pricing_map_only(pricing_frame_counter, coords, matrix, curr_node,
                                             nearest, explore, [curr_node], it, output_dir, formats,
                                             full_coord_span)
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
                                         nearest, explore, full_nodes, it, output_dir, formats,
                                         full_coord_span)

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

    NOTE: the `if coords is None or np.all(coords == 0)` MDS fallback
    below is now a redundant safety net, not the primary fix for
    degenerate coordinates -- see algo_data_loader.py's FIX LOG. The
    much more common case (only SOME nodes missing metadata, not all)
    is now handled inside AmazonDataLoader.extract_single_route() itself.
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
