"""
experiment_bf_qubit_scaling.py

Standalone experiment: take one or more Amazon routes, and run the FULL
receding-horizon heuristic algorithm (not an isolated subproblem -- the
actual end-to-end tour construction, algo_hybrid_bellmanford.py's
run_algo_hybrid_bf) at every window_size k from 2 up to k_max (default
20), to see what increasing the candidate-window size ("qubit count",
in QAOA terms) does to final tour quality -- and save a route-comparison
picture (Amazon vs. Bellman-Ford heuristic) for every k tested, on every
route.

--------------------------------------------------------------------------
WHY BELLMAN-FORD, NOT QAOA, FOR THIS EXPERIMENT
--------------------------------------------------------------------------
This is explicitly a THEORY experiment, not a claim about what QAOA
itself would achieve at k=20: real QAOA statevector simulation costs
2^(k^2), making k=20 (400 qubits) astronomically infeasible on any
classical simulator, and far beyond any current or near-term quantum
hardware too. Bellman-Ford/Held-Karp (cg_hybrid_bellmanford_sub.py's
solve_bellman_ford_subtour, reused here via
algo_hybrid_bellmanford.run_algo_hybrid_bf) is EXACT and costs only
O(2^k * k^2) -- still exponential, but with a base that makes k=20
tractable in minutes rather than never. Using it here answers a
DIFFERENT, well-posed question: "if the k-node sub-tour solver were
always exactly optimal (which QAOA is NOT, and doesn't get closer to by
adding qubits alone -- see algo_hybrid_LRWSQAOA.py's own FIX LOG,
'DEPTH CHANGE', for a directly relevant negative result), what would a
LARGER candidate window do to the final tour, holding the rest of the
receding-horizon construction fixed?" That's a genuinely useful
theoretical ceiling to know, even though no real QAOA circuit at k=20 is
attemptable -- it separates "what would more qubits even buy you if the
solver were perfect" from "does QAOA's own solve quality scale with
qubit count" (a question already answered, empirically, in the
negative, in algo_hybrid_LRWSQAOA.py's own revision history).

--------------------------------------------------------------------------
WHY batch_count = window_size FOR THIS EXPERIMENT SPECIFICALLY
--------------------------------------------------------------------------
Elsewhere in this codebase (run_experiment_ALL.py, algo_hybrid_
bellmanford.py's own typical usage), batch_count is usually much
smaller than the window size -- commit a few nodes, re-plan with a
fresh window, repeat. That means roughly (n-1)/batch_count solver calls
per route -- for batch_count=1 on a 150-stop route, ~150 calls. At
k=20, each Bellman-Ford call was measured (see cg_hybrid_bellmanford_
sub.py's own module docstring) at several tens of seconds -- 150 calls
at that cost would take HOURS for a single k value, and this experiment
sweeps 19 of them (k=2..20) across, by default, 3 routes.

This experiment instead sets batch_count = window_size: commit the
ENTIRE resolved k-node window before re-planning. That cuts solver
calls to roughly ceil((n-1)/k) -- for k=20 on a 150-stop route, about
8 calls, not 150. This is a deliberate, documented choice specific to
keeping this scaling SWEEP tractable, not a claim that batch_count=
window_size is how you'd want to run this algorithm for actual route
quality (a smaller batch_count with more frequent re-planning is
generally better for that, exactly why it's the default elsewhere).

CONSEQUENCE WORTH KNOWING: because batch_count changes WITH k here,
what you see on the resulting chart is not a clean isolation of "k
alone" -- larger k also means less frequent re-planning (fewer, bigger
commits), which can itself hurt quality independent of the solver's own
capability. If the chart looks jagged/non-monotonic rather than a clean
upward trend, that confound is a real, likely contributor, not
necessarily evidence that "more qubits don't help" on its own.

--------------------------------------------------------------------------
WHAT THIS EXPERIMENT DOES NOT COVER
--------------------------------------------------------------------------
Column generation (cg_hybrid_bellmanford_sub.py) is NOT swept here by
default. CG's pricing step calls the k-node solver O(n_iterations * n)
times per run (not O(n/k) like the heuristic here) -- even with
aggressively reduced n_iterations and max_pricing_nodes, sweeping CG
across k=2..20 would take substantially longer than the heuristic sweep
above. Pass --include-cg to attempt it anyway, with a small, explicit,
reduced iteration/pricing-node budget and a printed runtime warning
before it starts -- this is opt-in specifically because of that cost,
not because CG is unsupported.

--------------------------------------------------------------------------
MULTIPLE ROUTES (--num-routes, default 3)
--------------------------------------------------------------------------
Runs the same k=2..k_max sweep independently on `num_routes` different
routes (sampled the same way run_experiment_ALL.py samples them), so a
trend (or lack of one) can be checked across more than a single route's
idiosyncrasies. The combined chart plots one line per route per
algorithm; the CSV has one row per (route, algorithm, k). Runtime scales
roughly linearly with num_routes -- the pre-flight estimate printed
before running accounts for this.

--------------------------------------------------------------------------
ROUTE PICTURES
--------------------------------------------------------------------------
For every k tested, on every route, a route-comparison figure (Amazon
planned vs. Bellman-Ford heuristic at that k) is saved via the same
plot_publication.generate_overall_visualizations every other experiment
script in this project uses -- both with-depot and without-depot
variants. This means num_routes * (k_max-1) figures by default (e.g.
3 routes * 19 k-values = 57, x2 for the depot variants = 114 PNGs) --
use --plot-every-k to thin this out (e.g. --plot-every-k 3 plots only
every third k) or --no-route-plots to disable entirely.
"""

import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algo_hybrid_bellmanford import run_algo_hybrid_bf
from cg_hybrid_bellmanford_sub import run_cg_hybrid_bellmanford_sub


_ROUTE_COLORS = ["#1f4e79", "#b5651d", "#1a6b1a", "#6a3d9a", "#8c1c13", "#2c7873"]


def run_qubit_scaling_experiment(
    data, k_max=20, seed=2026,
    include_cg=False, cg_n_iterations=2, cg_max_pricing_nodes=8,
    generate_route_plots=True, plot_output_dir="./bf_qubit_scaling_results",
    plot_every_k=1, plot_formats=("png",),
):
    """
    For ONE route (`data`, with an 'amazon_planned_tour' key for the
    baseline comparison), runs the full heuristic (run_algo_hybrid_bf)
    at every window_size k = 2 .. min(n-1, k_max), holding
    exploration_percent=0 (deterministic) and batch_count=window_size
    (see module docstring for why). Returns a list of per-k result
    dicts (each tagged with this route's id). If generate_route_plots,
    also saves an Amazon-vs-heuristic route-comparison figure for every
    k tested (subject to plot_every_k thinning) via
    plot_publication.generate_overall_visualizations.

    If include_cg=True, ALSO runs run_cg_hybrid_bellmanford_sub at each
    k with a reduced (cg_n_iterations, cg_max_pricing_nodes) budget,
    and plots its route too.
    """
    from algo_data_loader import compute_open_route_cost

    matrix = data["matrix"]
    n = data["n_nodes"]
    route_id = data.get("route_id", "unknown")
    amazon_tour = data.get("amazon_planned_tour")
    amazon_cost = compute_open_route_cost(amazon_tour, matrix) if amazon_tour is not None else None

    k_max_effective = min(n - 1, k_max)
    k_values = list(range(2, k_max_effective + 1))

    print(f"=== Qubit-count (window_size) scaling experiment: route {route_id} ({n} stops) ===")
    if amazon_cost is not None:
        print(f"Amazon planned cost: {amazon_cost:,.2f}")
    if k_max_effective < k_max:
        print(f"NOTE: route only has {n-1} non-depot stops; testing k=2..{k_max_effective} "
              f"instead of the requested k_max={k_max}.")
    print(f"Testing k = 2..{k_max_effective}, batch_count=window_size (see module docstring for why), "
          f"exploration_percent=0 (deterministic).\n")

    if generate_route_plots:
        from plot_publication import generate_overall_visualizations
        route_plot_dir = os.path.join(plot_output_dir, "route_plots", str(route_id))
        os.makedirs(route_plot_dir, exist_ok=True)

    results = []
    for k in k_values:
        t0 = time.time()
        res = run_algo_hybrid_bf(
            data, window_size=k, batch_count=k, exploration_percent=0.0,
            seed=seed, bf_max_k=k_max_effective,
        )
        elapsed = time.time() - t0
        cost = res["cost"]
        improvement_pct = (
            -100.0 * (cost - amazon_cost) / amazon_cost if amazon_cost else None
        )
        num_calls = -(-(n - 1) // k)  # ceil((n-1)/k)

        row = {
            "route_id": route_id, "k": k, "cost": cost, "improvement_pct_vs_amazon": improvement_pct,
            "runtime_sec": elapsed, "num_solver_calls": num_calls, "algorithm": "heuristic_bf",
        }
        results.append(row)

        imp_str = f"{improvement_pct:+6.2f}%" if improvement_pct is not None else "   n/a "
        print(f"k={k:2d}: cost={cost:10.2f}  vs Amazon={imp_str}  "
              f"time={elapsed:8.3f}s  solver_calls~{num_calls}")

        if generate_route_plots and amazon_tour is not None and (k - 2) % plot_every_k == 0:
            generate_overall_visualizations(
                data, res["tour"], cost, f"hbf_k{k}", route_plot_dir,
                algo_label=f"Heuristic (Bellman-Ford), k={k}", algo_color="#b5651d",
                formats=plot_formats,
            )

        if include_cg:
            t0 = time.time()
            try:
                cg_res = run_cg_hybrid_bellmanford_sub(
                    data, window_size=k, exploration_percent=0.0, seed=seed, bf_max_k=k_max_effective,
                    n_iterations=cg_n_iterations, max_pricing_nodes=min(cg_max_pricing_nodes, n),
                )
                cg_elapsed = time.time() - t0
                cg_cost = cg_res["cost"]
                cg_improvement_pct = (
                    -100.0 * (cg_cost - amazon_cost) / amazon_cost if amazon_cost else None
                )
                cg_row = {
                    "route_id": route_id, "k": k, "cost": cg_cost,
                    "improvement_pct_vs_amazon": cg_improvement_pct, "runtime_sec": cg_elapsed,
                    "num_solver_calls": cg_n_iterations * min(cg_max_pricing_nodes, n), "algorithm": "cg_bf",
                }
                results.append(cg_row)
                cg_imp_str = f"{cg_improvement_pct:+6.2f}%" if cg_improvement_pct is not None else "   n/a "
                print(f"      [cg_bf, n_iterations={cg_n_iterations}, max_pricing_nodes="
                      f"{min(cg_max_pricing_nodes, n)}]: cost={cg_cost:10.2f}  "
                      f"vs Amazon={cg_imp_str}  time={cg_elapsed:8.3f}s")

                if generate_route_plots and amazon_tour is not None and (k - 2) % plot_every_k == 0:
                    generate_overall_visualizations(
                        data, cg_res["tour"], cg_cost, f"cgbf_k{k}", route_plot_dir,
                        algo_label=f"CG (Bellman-Ford), k={k}", algo_color="#8c1c13",
                        formats=plot_formats,
                    )
            except Exception as e:
                warnings.warn(f"cg_bf failed at k={k}: {e}")

    return results


def run_qubit_scaling_experiment_multi(routes_data, **kwargs):
    """Runs run_qubit_scaling_experiment independently on each route in
    `routes_data`, tagging every result row with its route_id (already
    done inside the single-route function), and returns the combined
    list of all routes' rows."""
    all_results = []
    for i, data in enumerate(routes_data, 1):
        print(f"\n{'='*70}\nRoute {i}/{len(routes_data)}\n{'='*70}")
        all_results.extend(run_qubit_scaling_experiment(data, **kwargs))
    return all_results


def plot_scaling_results(results, output_path, title_suffix=""):
    """
    Line chart(s): cost and improvement-vs-Amazon, both against k. One
    line per (route_id, algorithm) combination present in `results` --
    so with multiple routes, each route gets its own color, and
    heuristic_bf/cg_bf (if present) are distinguished by line style
    (solid vs. dashed) within a route's color rather than a fully
    separate color, keeping the legend readable with multiple routes.
    Saved as PNG.
    """
    route_ids = sorted({r["route_id"] for r in results})
    route_color = {rid: _ROUTE_COLORS[i % len(_ROUTE_COLORS)] for i, rid in enumerate(route_ids)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor("#F7F8FA")
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", linewidth=0.6, color="#E4E6EA", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    any_improvement_data = False
    for rid in route_ids:
        for algo, style, marker in [("heuristic_bf", "-", "o"), ("cg_bf", "--", "s")]:
            rows = sorted([r for r in results if r["route_id"] == rid and r["algorithm"] == algo],
                          key=lambda r: r["k"])
            if not rows:
                continue
            color = route_color[rid]
            label = f"{rid} ({'Heuristic' if algo == 'heuristic_bf' else 'CG'})"
            k_vals = [r["k"] for r in rows]
            cost_vals = [r["cost"] for r in rows]
            ax1.plot(k_vals, cost_vals, style, marker=marker, color=color, label=label, markersize=4)
            if rows[0]["improvement_pct_vs_amazon"] is not None:
                any_improvement_data = True
                imp_vals = [r["improvement_pct_vs_amazon"] for r in rows]
                ax2.plot(k_vals, imp_vals, style, marker=marker, color=color, label=label, markersize=4)

    ax1.set_xlabel("k (window_size / theoretical qubit count)")
    ax1.set_ylabel("Tour cost")
    ax1.set_title("Final tour cost vs. k")
    ax1.legend(fontsize=8, loc="best")

    if any_improvement_data:
        ax2.axhline(0, color="#9A9EA6", linewidth=1, linestyle=":")
        ax2.set_xlabel("k (window_size / theoretical qubit count)")
        ax2.set_ylabel("Improvement vs. Amazon (%)")
        ax2.set_title("Improvement vs. Amazon planned, vs. k")
        ax2.legend(fontsize=8, loc="best")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No Amazon baseline available", ha="center", va="center",
                  transform=ax2.transAxes, color="#7A7E86")

    fig.suptitle(f"Theoretical effect of increasing k{title_suffix}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_route_sample(data_dir, num_routes, seed, synthetic_n):
    """Loads `num_routes` real Amazon routes if the dataset is found,
    else that many synthetic routes (nearest-neighbor tour standing in
    for 'Amazon planned', consistent with this file's single-route
    fallback convention elsewhere)."""
    routes = []
    if os.path.exists(data_dir) or os.path.exists("./data"):
        try:
            from algo_data_loader import AmazonDataLoader

            real_dir = data_dir if os.path.exists(data_dir) else "./data"
            loader = AmazonDataLoader(data_dir=real_dir)
            if loader.travel_times:
                all_route_ids = sorted(loader.travel_times.keys())
                rng0 = np.random.default_rng(seed)
                n_pick = min(num_routes, len(all_route_ids))
                chosen_ids = rng0.choice(all_route_ids, size=n_pick, replace=False).tolist()
                for route_id in chosen_ids:
                    extracted = loader.extract_single_route(route_id)
                    matrix = np.array(extracted["matrix"])
                    coords = np.array(extracted["coords"])
                    if coords is None or np.all(coords == 0):
                        from sklearn.manifold import MDS
                        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed)
                        coords = mds.fit_transform(matrix)
                    routes.append({
                        "route_id": extracted.get("route_id", route_id),
                        "n_nodes": extracted["n_nodes"], "coords": coords, "matrix": matrix,
                        "depot_idx": extracted.get("depot_idx", 0),
                        "amazon_planned_tour": extracted["amazon_planned_sequence"],
                    })
                print(f"Loaded {len(routes)} real route(s) from '{real_dir}'.\n")
                return routes
        except Exception as e:
            print(f"Could not load real Amazon routes ({e}); falling back to synthetic routes.\n")

    rng0 = np.random.default_rng(seed)
    for i in range(num_routes):
        n = synthetic_n
        coords = rng0.uniform(0, 100, size=(n, 2))
        matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        synth_tour = [0]
        remaining = set(range(1, n))
        while remaining:
            last = synth_tour[-1]
            nxt = min(remaining, key=lambda x: matrix[last, x])
            synth_tour.append(nxt)
            remaining.remove(nxt)
        routes.append({
            "route_id": f"synthetic-{n}stops-seed{seed}-{i}",
            "n_nodes": n, "coords": coords, "matrix": matrix, "depot_idx": 0,
            "amazon_planned_tour": synth_tour,
        })
    print(f"No Amazon dataset found at '{data_dir}' -- using {num_routes} synthetic route(s) "
          f"instead (seed={seed}), with nearest-neighbor tours standing in for 'Amazon planned'.\n")
    return routes


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(
        description="Runs the full Bellman-Ford heuristic on one or more Amazon routes at every "
                    "window_size k=2..k_max, to show the theoretical effect of increasing "
                    "candidate-window size ('qubit count') on final tour quality -- see module "
                    "docstring for why Bellman-Ford (exact) stands in for 'what would happen if "
                    "QAOA scaled perfectly', which QAOA itself does not."
    )
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training")
    parser.add_argument("--num-routes", type=int, default=3,
                        help="How many routes to sample and sweep independently (default 3).")
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--synthetic-n", type=int, default=60,
                        help="Size of each synthetic fallback route if no real dataset is found.")
    parser.add_argument("--output-dir", type=str, default="./bf_qubit_scaling_results")
    parser.add_argument("--include-cg", action="store_true",
                        help="ALSO sweep cg_bf at each k (reduced iteration/pricing-node budget -- "
                             "see module docstring for why this is opt-in and slower).")
    parser.add_argument("--cg-n-iterations", type=int, default=2)
    parser.add_argument("--cg-max-pricing-nodes", type=int, default=8)
    parser.add_argument("--no-route-plots", action="store_true",
                        help="Disable saving Amazon-vs-heuristic route-comparison pictures per k.")
    parser.add_argument("--plot-every-k", type=int, default=1,
                        help="Only save route pictures every N k values (default 1 = every k). "
                             "Use to thin out output volume for large sweeps.")
    parser.add_argument("--plot-formats", type=str, nargs="+", default=["png"], choices=["png", "pdf"])
    parser.add_argument("--yes", action="store_true", help="Skip the pre-flight confirmation prompt.")

    args = parser.parse_args()

    routes_data = _load_route_sample(args.data_dir, args.num_routes, args.seed, args.synthetic_n)

    total_est_calls = 0
    for data in routes_data:
        n = data["n_nodes"]
        k_max_eff = min(n - 1, args.k_max)
        total_est_calls += -(-(n - 1) // k_max_eff)
    print(
        f"Pre-flight estimate: sweeping k=2..{args.k_max} across {len(routes_data)} route(s). "
        f"The most expensive single k per route needs a handful of exact Bellman-Ford solver "
        f"calls, each of which can take on the order of tens of seconds at k~18-20 (see "
        f"cg_hybrid_bellmanford_sub.py's module docstring for measured numbers) -- total runtime "
        f"scales roughly linearly with --num-routes ({len(routes_data)} here), typically several "
        f"minutes per route.\n"
    )
    if not args.yes:
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            resp = "y"
        if resp != "y":
            print("Aborted.")
            raise SystemExit(0)

    results = run_qubit_scaling_experiment_multi(
        routes_data, k_max=args.k_max, seed=args.seed, include_cg=args.include_cg,
        cg_n_iterations=args.cg_n_iterations, cg_max_pricing_nodes=args.cg_max_pricing_nodes,
        generate_route_plots=not args.no_route_plots, plot_output_dir=args.output_dir,
        plot_every_k=max(1, args.plot_every_k), plot_formats=tuple(args.plot_formats),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "scaling_multi_route.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["route_id", "algorithm", "k", "cost",
                                                "improvement_pct_vs_amazon", "runtime_sec",
                                                "num_solver_calls"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults saved to: {csv_path}")

    plot_path = os.path.join(args.output_dir, "scaling_multi_route.png")
    plot_scaling_results(results, plot_path, title_suffix=f" ({len(routes_data)} routes)")
    print(f"Chart saved to: {plot_path}")
    if not args.no_route_plots:
        print(f"Route-comparison pictures saved under: {os.path.join(args.output_dir, 'route_plots')}/")
