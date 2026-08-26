"""
plot_publication_all.py

Three-way route visualizations for the unified experiment
(run_experiment_ALL.py): Amazon Planned vs. Heuristic (Algorithm 1,
WS-LR-QAOA) vs. CG/TDE-QP (Algorithm 2), side by side in one figure.

This module does NOT redefine any of the drawing primitives that
plot_publication.py already owns (route styling, arrowheads, start/end
markers, sparse labeling, PNG+PDF export) -- it imports and reuses them,
so the 2-panel figures produced by run_amazon_experiment.py /
run_CG_experiment.py and the 3-panel figures produced here stay visually
identical in every respect except panel count. If you ever change route
styling, change it once in plot_publication.py and both call sites pick
it up.

Public entry point:
    generate_three_way_visualization(
        data, heuristic_tour, heuristic_cost, heuristic_param_str,
        cg_tour, cg_cost, cg_param_str, output_dir, ...
    )
Writes into `<output_dir>/plots_with_depot/` and
`<output_dir>/plots_without_depot/` -- pass
`<experiment_output_dir>/visualise_experiments_ALL` as `output_dir` from
run_experiment_ALL.py so everything lands under that named folder, per
spec.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch

from plot_publication import (
    _route_panel,
    _save_all_formats,
    COLOR_BG,
    CMAP_AMAZON,
    CMAP_HYBRID,
)

# A third color identity, distinct from Amazon's blue (#1f4e79) and the
# heuristic's default green (#1a6b1a), for the CG panel.
COLOR_CG = "#6a3d9a"          # purple, matches run_CG_experiment.py's algo_color
COLOR_WIN = "#1a7a3c"
COLOR_LOSE = "#b0392f"
COLOR_NEUTRAL = "#5A5E66"


def _draw_three_way_scorecard(
    fig, route_id, n_stops,
    amazon_cost, heuristic_cost, heuristic_label, heuristic_param_str,
    cg_cost, cg_label, cg_param_str,
):
    """
    Banner strip summarizing all three costs at a glance plus two
    improvement badges (Heuristic vs. Amazon, CG vs. Amazon) and a
    third, small head-to-head badge (Heuristic vs. CG) so the reader
    does not have to do that subtraction themselves.
    """
    def pct_improve(other_cost, base_cost):
        return -100.0 * (other_cost - base_cost) / base_cost if base_cost else 0.0

    heuristic_vs_amazon = pct_improve(heuristic_cost, amazon_cost)
    cg_vs_amazon = pct_improve(cg_cost, amazon_cost)
    heuristic_vs_cg = pct_improve(heuristic_cost, cg_cost)  # >0 => heuristic cheaper than CG

    ax = fig.add_axes([0.04, 0.855, 0.92, 0.085])
    ax.axis("off")
    box = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                          boxstyle="round,pad=0,rounding_size=0.10",
                          linewidth=1.0, edgecolor="#DDE0E4", facecolor="#FFFFFF", zorder=1)
    ax.add_patch(box)

    ax.text(0.02, 0.68, f"Route {route_id}", transform=ax.transAxes,
            ha="left", va="center", fontsize=13, fontweight="bold", color="#26282B")
    ax.text(0.02, 0.20, f"{n_stops} stops", transform=ax.transAxes,
            ha="left", va="center", fontsize=9, color="#7A7E86")

    # Three raw costs, aligned under their respective panel.
    ax.text(0.19, 0.68, f"Amazon: {amazon_cost:,.0f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=11.5, color="#1f4e79")
    ax.text(0.19, 0.22, "baseline", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="#9A9EA6")

    ax.text(0.47, 0.68, f"{heuristic_label}: {heuristic_cost:,.0f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=11.5, color="#1a6b1a")
    h_color = COLOR_WIN if heuristic_vs_amazon > 0 else COLOR_LOSE
    ax.text(0.47, 0.22, f"{heuristic_vs_amazon:+.1f}% vs Amazon", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.5, color=h_color, fontweight="bold")
    ax.text(0.47, 0.02, heuristic_param_str, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.8, color="#B0B3B8")

    ax.text(0.75, 0.68, f"{cg_label}: {cg_cost:,.0f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=11.5, color=COLOR_CG)
    c_color = COLOR_WIN if cg_vs_amazon > 0 else COLOR_LOSE
    ax.text(0.75, 0.22, f"{cg_vs_amazon:+.1f}% vs Amazon", transform=ax.transAxes,
            ha="center", va="center", fontsize=8.5, color=c_color, fontweight="bold")
    ax.text(0.75, 0.02, cg_param_str, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.8, color="#B0B3B8")

    # Head-to-head badge: which of the two algorithms wins this route.
    h2h_color = COLOR_WIN if heuristic_vs_cg > 0 else (COLOR_LOSE if heuristic_vs_cg < 0 else COLOR_NEUTRAL)
    h2h_text = (
        f"Heuristic \u25B2 {abs(heuristic_vs_cg):.1f}%" if heuristic_vs_cg > 0
        else (f"CG \u25B2 {abs(heuristic_vs_cg):.1f}%" if heuristic_vs_cg < 0 else "Tie")
    )
    badge = FancyBboxPatch((0.905, 0.14), 0.085, 0.72, transform=ax.transAxes,
                            boxstyle="round,pad=0,rounding_size=0.15",
                            linewidth=0, facecolor=h2h_color, alpha=0.14, zorder=2)
    ax.add_patch(badge)
    ax.text(0.9475, 0.68, "Head-to-head", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.5, color="#7A7E86")
    ax.text(0.9475, 0.32, h2h_text, transform=ax.transAxes,
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=h2h_color,
            linespacing=1.3, zorder=3)


def _three_panel_figure(
    coords, amazon_path, heuristic_path, cg_path,
    depot_idx, amazon_cost, heuristic_cost, heuristic_label, cg_cost, cg_label,
    max_node_labels, suptitle,
):
    fig = plt.figure(figsize=(28, 10.5))
    fig.patch.set_facecolor(COLOR_BG)
    gs = fig.add_gridspec(1, 3, left=0.035, right=0.98, top=0.78, bottom=0.06, wspace=0.12)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    _route_panel(axes[0], coords, amazon_path, CMAP_AMAZON, "#1f4e79",
                 f"Amazon Planned \u2014 {suptitle}", amazon_cost, depot_idx=depot_idx,
                 max_labels=max_node_labels)
    _route_panel(axes[1], coords, heuristic_path, CMAP_HYBRID, "#1a6b1a",
                 f"{heuristic_label} \u2014 {suptitle}", heuristic_cost, depot_idx=depot_idx,
                 max_labels=max_node_labels)
    _route_panel(axes[2], coords, cg_path, "Purples", COLOR_CG,
                 f"{cg_label} \u2014 {suptitle}", cg_cost, depot_idx=depot_idx,
                 max_labels=max_node_labels)

    fig.suptitle(f"Route Comparison: Amazon vs. Heuristic vs. CG ({suptitle})", fontsize=18, y=0.975)
    return fig


def generate_three_way_visualization(
    data,
    heuristic_tour, heuristic_cost, heuristic_param_str,
    cg_tour, cg_cost, cg_param_str,
    output_dir,
    max_node_labels=30, formats=("png", "pdf"),
    heuristic_label="Heuristic (WS-LR-QAOA)", cg_label="CG (TDE-QP)",
):
    """
    Writes both the with-depot and without-depot 3-panel comparison
    figures for one route into `output_dir`. Mirrors
    plot_publication.generate_overall_visualizations's folder layout
    (plots_with_depot / plots_without_depot) one level under whatever
    `output_dir` the caller passes -- run_experiment_ALL.py passes
    `<experiment_output_dir>/visualise_experiments_ALL`.
    """
    from algo_data_loader import compute_open_route_cost  # single source of truth

    depot_dir = os.path.join(output_dir, "plots_with_depot")
    no_depot_dir = os.path.join(output_dir, "plots_without_depot")
    os.makedirs(depot_dir, exist_ok=True)
    os.makedirs(no_depot_dir, exist_ok=True)

    route_id = data["route_id"]
    coords = data["coords"]
    matrix = data["matrix"]
    depot_idx = data["depot_idx"]
    amazon_tour = data["amazon_planned_tour"]
    amazon_cost = compute_open_route_cost(amazon_tour, matrix)
    n_stops = len(coords)

    filename_slug = f"{route_id}__{heuristic_param_str}__{cg_param_str}"

    # ------------------ 1. WITH DEPOT ------------------
    fig = _three_panel_figure(
        coords, amazon_tour, heuristic_tour, cg_tour, depot_idx,
        amazon_cost, heuristic_cost, heuristic_label, cg_cost, cg_label,
        max_node_labels, suptitle="With Depot",
    )
    _draw_three_way_scorecard(
        fig, route_id, n_stops, amazon_cost, heuristic_cost, heuristic_label,
        heuristic_param_str, cg_cost, cg_label, cg_param_str,
    )
    _save_all_formats(fig, os.path.join(depot_dir, filename_slug), formats)
    plt.close(fig)

    # ------------------ 2. WITHOUT DEPOT ------------------
    amazon_nd = [i for i in amazon_tour if i != depot_idx]
    heuristic_nd = [i for i in heuristic_tour if i != depot_idx]
    cg_nd = [i for i in cg_tour if i != depot_idx]

    fig = _three_panel_figure(
        coords, amazon_nd, heuristic_nd, cg_nd, None,
        amazon_cost, heuristic_cost, heuristic_label, cg_cost, cg_label,
        max_node_labels, suptitle="Delivery Stops Only",
    )
    _draw_three_way_scorecard(
        fig, route_id, n_stops - 1, amazon_cost, heuristic_cost, heuristic_label,
        heuristic_param_str, cg_cost, cg_label, cg_param_str,
    )
    _save_all_formats(fig, os.path.join(no_depot_dir, filename_slug), formats)
    plt.close(fig)

    plt.close("all")
