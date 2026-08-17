"""
plot_publication.py

Publication-quality route visualizations for Amazon Planned vs. Hybrid
Algo 2+5 comparisons. Drop-in replacement for
`generate_overall_visualizations()` in run_amazon_experiment.py -- same
name, same signature, same two output folders (plots_with_depot /
plots_without_depot), so switching to it is a one-line import change:

    from plot_publication import generate_overall_visualizations

Design choices, and why:
  - Route direction is shown as a single solid, fully-opaque color line
    (no gradient/opacity fade -- an earlier version faded brightness
    along the path, which read as "parts of the route are invisible" on
    real, denser routes) plus a handful of arrowheads along its length.
  - Start and end stops get distinct markers (triangle / square) so the
    route's direction is legible even in black-and-white print.
  - Node-index labels are a big source of visual noise once a route has
    more than ~25-30 stops (which real Amazon routes routinely do -- often
    100-250). Past that count, labels are sparsely subsampled rather than
    drawn for every node, and depot/start/end are always labeled
    regardless of count.
  - Equal aspect ratio is enforced so route shape isn't visually distorted
    by unequal lat/lng axis scaling.
  - Every figure is saved as both PNG (raster, presentation/slides) and
    PDF (vector, camera-ready for a paper) at once.
  - A single shared style function keeps the two panels (Amazon vs.
    Hybrid) visually consistent -- same fonts, same marker sizes, same
    spine/grid treatment -- so the only thing that differs between panels
    is the actual route data, which is the point of the comparison.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# Shared visual language
# ---------------------------------------------------------------------

COLOR_DEPOT = "#C1272D"       # red diamond
COLOR_GRID = "#D8D8D8"
COLOR_TEXT = "#2B2B2B"
CMAP_AMAZON = "Blues"
CMAP_HYBRID = "Greens"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _style_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#555555", labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.6, color=COLOR_GRID, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")


def _plot_directional_route(ax, path_coords, color, linewidth=1.5, zorder=3, n_arrows=6):
    """
    Draws `path_coords` (an (N,2) array in visit order) as a single flat,
    fully-opaque, solid-color line -- no color/opacity gradient. Direction
    of travel is shown ONLY via a handful of arrowheads along the path.

    FIX (round 2): a previous version faded brightness from t=0.15 to
    0.95 along a colormap, which made the start of routes look faint or
    invisible. A narrower fade (0.55-1.0) was tried next, but any
    gradient at all still reads as "parts of the line have different
    opacity" on real, denser routes -- which is exactly what was flagged
    as still wrong. There is now no gradient of any kind: one solid
    color, drawn with a thin (not thick) linewidth per request, and
    arrowheads for direction instead of a brightness cue.
    """
    if len(path_coords) < 2:
        return None

    ax.plot(
        path_coords[:, 0], path_coords[:, 1], "-",
        color=color, linewidth=linewidth, alpha=1.0,
        solid_capstyle="round", solid_joinstyle="round", zorder=zorder,
    )

    # Direction arrows: a handful of fixed-size arrowheads along the path,
    # scaled to the plot's own extent so they look consistent regardless
    # of the route's coordinate range. Same solid color as the line.
    pts = path_coords.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n_seg = len(segments)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    diag = float(np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    arrow_len = diag * 0.028

    idxs = sorted(set(np.linspace(0, n_seg - 1, min(n_arrows, n_seg)).astype(int).tolist()))
    for i in idxs:
        p0, p1 = segments[i]
        direction = p1 - p0
        norm = float(np.hypot(*direction))
        if norm < 1e-9:
            continue
        unit = direction / norm
        mid = (p0 + p1) / 2.0
        tail = mid - unit * arrow_len / 2.0
        head = mid + unit * arrow_len / 2.0
        ax.annotate(
            "", xy=tuple(head), xytext=tuple(tail),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=13),
            zorder=zorder + 1,
        )
    return None


def _mark_start_end(ax, path_coords, color, zorder=6):
    if len(path_coords) < 1:
        return
    ax.scatter(*path_coords[0], marker="^", s=190, c=color, edgecolors="white",
               linewidths=1.2, zorder=zorder, label="_nolegend_")
    if len(path_coords) > 1:
        ax.scatter(*path_coords[-1], marker="s", s=155, c=color, edgecolors="white",
                   linewidths=1.2, zorder=zorder, label="_nolegend_")


def _annotate_sparse(ax, coords, indices, max_labels=30, fontsize=8, always_include=()):
    """
    Labels every node if there are few enough of them to stay readable;
    otherwise evenly subsamples down to ~max_labels, always keeping
    `always_include` (e.g. depot/start/end) regardless of count.
    """
    indices = list(indices)
    always = set(always_include) & set(indices)
    rest = [i for i in indices if i not in always]

    if len(indices) <= max_labels:
        chosen = indices
    else:
        budget = max(0, max_labels - len(always))
        step = max(1, len(rest) // max(budget, 1))
        chosen = list(always) + rest[::step][:budget]

    for idx in chosen:
        ax.annotate(
            str(idx), (coords[idx, 0], coords[idx, 1]),
            fontsize=fontsize, color=COLOR_TEXT, xytext=(3, 3),
            textcoords="offset points", zorder=7,
        )


def _route_panel(ax, coords, path_indices, cmap_name, route_color, title, cost,
                  depot_idx=None, max_labels=30):
    # NOTE: cmap_name is accepted but no longer used -- _plot_directional_route
    # draws a solid `route_color` line now, not a colormap gradient (see its
    # docstring). Kept in the signature only so existing callers that pass
    # algo_cmap/CMAP_AMAZON/CMAP_HYBRID by keyword don't break; it has no
    # effect on the rendered output.
    path_coords = coords[path_indices]
    _plot_directional_route(ax, path_coords, route_color)
    _mark_start_end(ax, path_coords, route_color)

    if depot_idx is not None and depot_idx in path_indices:
        ax.scatter(coords[depot_idx, 0], coords[depot_idx, 1], c=COLOR_DEPOT,
                   s=230, marker="D", edgecolors="white", linewidths=1.3,
                   zorder=8, label="Depot")

    always = {path_indices[0], path_indices[-1]}
    if depot_idx is not None:
        always.add(depot_idx)
    _annotate_sparse(ax, coords, path_indices, max_labels=max_labels, always_include=always)

    ax.set_title(f"{title}\nCost: {cost:.2f}", fontsize=13)
    _style_axes(ax)

    legend_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=route_color,
               markeredgecolor="white", markersize=9, label="Start"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=route_color,
               markeredgecolor="white", markersize=8, label="End"),
    ]
    if depot_idx is not None and depot_idx in path_indices:
        legend_handles.append(
            Line2D([0], [0], marker="D", color="none", markerfacecolor=COLOR_DEPOT,
                   markeredgecolor="white", markersize=9, label="Depot")
        )
    ax.legend(handles=legend_handles, loc="best")


def _save_all_formats(fig, path_no_ext, formats=("png", "pdf")):
    for ext in formats:
        fig.savefig(f"{path_no_ext}.{ext}")


def generate_overall_visualizations(
    data, hybrid_tour, hybrid_cost, param_str, output_dir,
    max_node_labels=30, formats=("png", "pdf"),
    algo_label="Hybrid Algo 2+5", algo_cmap=CMAP_HYBRID, algo_color="#1a6b1a",
):
    """
    Publication-quality side-by-side comparison plots between Amazon
    Planned and a second algorithm's tour. Same signature/output layout
    as the original: writes into `plots_with_depot/` and
    `plots_without_depot/` under `output_dir`.

    `algo_label`/`algo_color` identify the second algorithm in titles and
    give it its own color identity -- default matches the original
    Hybrid Algo 2+5 look exactly, so existing callers (e.g.
    run_amazon_experiment.py) are unaffected. Other callers (e.g.
    run_CG_experiment.py) should pass their own algo_label/algo_color so
    the right panel is correctly labeled instead of silently showing
    "Hybrid Algo 2+5" for a different algorithm's results. algo_cmap is
    accepted for backward compatibility but no longer affects rendering
    (see _route_panel) -- only algo_color does.
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

    filename_slug = f"{route_id}_{param_str}"

    # ------------------ 1. WITH DEPOT ------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8))
    _route_panel(axes[0], coords, amazon_tour, CMAP_AMAZON, "#1f4e79",
                 "Amazon Planned (With Depot)", amazon_cost, depot_idx=depot_idx,
                 max_labels=max_node_labels)
    _route_panel(axes[1], coords, hybrid_tour, algo_cmap, algo_color,
                 f"{algo_label} (With Depot)", hybrid_cost, depot_idx=depot_idx,
                 max_labels=max_node_labels)

    improvement_pct = -100.0 * (hybrid_cost - amazon_cost) / amazon_cost if amazon_cost else 0.0
    fig.suptitle(
        f"Route {route_id}  \u2022  {n_stops} stops  \u2022  "
        f"{algo_label + ' faster' if improvement_pct > 0 else 'Amazon faster'} by "
        f"{abs(improvement_pct):.1f}%\nParameters: {param_str}",
        fontsize=13, y=0.99,
    )
    fig.subplots_adjust(top=0.82)
    _save_all_formats(fig, os.path.join(depot_dir, filename_slug), formats)
    plt.close(fig)

    # ------------------ 2. WITHOUT DEPOT ------------------
    amazon_no_depot = [i for i in amazon_tour if i != depot_idx]
    hybrid_no_depot = [i for i in hybrid_tour if i != depot_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8))
    _route_panel(axes[0], coords, amazon_no_depot, CMAP_AMAZON, "#1f4e79",
                 "Amazon Planned (Delivery Stops Only)", amazon_cost, depot_idx=None,
                 max_labels=max_node_labels)
    _route_panel(axes[1], coords, hybrid_no_depot, algo_cmap, algo_color,
                 f"{algo_label} (Delivery Stops Only)", hybrid_cost, depot_idx=None,
                 max_labels=max_node_labels)

    fig.suptitle(
        f"Route {route_id}  \u2022  {n_stops - 1} delivery stops (depot excluded)\n"
        f"Parameters: {param_str}",
        fontsize=13, y=0.99,
    )
    fig.subplots_adjust(top=0.82)
    _save_all_formats(fig, os.path.join(no_depot_dir, filename_slug), formats)
    plt.close(fig)

    plt.close("all")


if __name__ == "__main__":
    # Minimal smoke test with synthetic data, so this can be sanity-checked
    # without the real Amazon dataset.
    rng = np.random.default_rng(0)
    n = 60
    coords = rng.uniform(0, 100, size=(n, 2))
    matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    depot_idx = 0
    amazon_tour = [depot_idx] + [i for i in range(1, n)]
    rng.shuffle(amazon_tour[1:])
    hybrid_tour = [depot_idx] + list(rng.permutation(np.arange(1, n)))

    data = {
        "route_id": "SMOKE_TEST",
        "coords": coords,
        "matrix": matrix,
        "depot_idx": depot_idx,
        "amazon_planned_tour": amazon_tour,
    }

    def compute_open_route_cost(tour, matrix):
        return float(sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

    hybrid_cost = compute_open_route_cost(hybrid_tour, matrix)

    import sys
    sys.modules["algo_data_loader"] = type(sys)("algo_data_loader")
    sys.modules["algo_data_loader"].compute_open_route_cost = compute_open_route_cost

    generate_overall_visualizations(data, hybrid_tour, hybrid_cost, "q4_exp0_b1_xy0", "/tmp/plot_smoke_test")
    print("Smoke test plots written to /tmp/plot_smoke_test/")
