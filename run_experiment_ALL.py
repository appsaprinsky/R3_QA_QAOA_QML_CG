"""
run_experiment_ALL.py

Unified experimental runner. For the SAME route sample (same
`--data-dir`/`--num-routes`/`--seed`, drawn by the identical sampling
function both standalone scripts already used), this script runs BOTH:

  - Algorithm 1: WS-LR-QAOA receding-horizon heuristic
    (run_algo_hybrid_2_5, algo_hybrid_LRWSQAOA.py)
  - Algorithm 2: Column Generation with Truncated, Dual-Embedded QAOA
    Pricing (run_cg_hybrid_lrwsqaoa_sub, cg_hybrid_lrwsqaoa_sub.py)

against the same Amazon Planned baseline, and writes ONE combined set of
comparison artifacts instead of two separate ones living in two
separate output directories:

    <output-dir>/
      experiment_ALL_detailed_results.csv        one row per
        (route, algorithm, parameter combination) -- long format, an
        "algorithm" column ("heuristic" / "cg") distinguishes rows;
        columns that don't apply to a given algorithm (e.g.
        batch_count for cg rows, only_improving_columns for heuristic
        rows) are left blank on that row rather than silently reused.
      experiment_ALL_summary_by_parameters.csv   grouped by algorithm
        + that algorithm's own parameter columns: mean cost, mean
        improvement % vs Amazon, win rate %, mean runtime, plus the
        cg-only diagnostics (segments selected, iterations run) where
        algorithm == "cg".
      experiment_ALL_route_comparison.csv        ONE row per route:
        Amazon cost side by side with the BEST (lowest-cost) Heuristic
        result and the BEST CG result found across each algorithm's own
        grid for that route, their runtimes, their param strings, each
        one's improvement % over Amazon, AND a direct Heuristic-vs-CG
        head-to-head % -- this is the single file to open for "did
        quantum-hybrid beat Amazon, and which of the two algorithms was
        better, per route."
      visualise_experiments_ALL/
        heuristic_vs_amazon/
          plots_with_depot/       Amazon vs. Heuristic, one figure per route
          plots_without_depot/    same, depot excluded
        cg_vs_amazon/
          plots_with_depot/       Amazon vs. CG, one figure per route
          plots_without_depot/    same, depot excluded
        (all PNG + PDF, via plot_publication.generate_overall_visualizations
        -- the SAME 2-panel figure function run_amazon_experiment.py and
        run_CG_experiment.py already use, called once per algorithm per
        route rather than folded into a single combined figure. An
        earlier version of this script drew one 3-panel Amazon|Heuristic|
        CG figure; that made both algorithms' routes harder to read at
        once and is no longer how this script plots.)

Design notes
------------
* Neither algorithm's own logic is reimplemented here -- both run_*
  functions, the cost metric, and the route-sampling function are
  imported from their single source of truth, exactly as the two
  standalone scripts already did. The route-sampling function itself
  was previously copy-pasted identically into BOTH
  run_amazon_experiment.py and run_CG_experiment.py; it is defined
  ONCE here instead, removing that duplication for the unified run.
* The two algorithms have different parameter grids (Heuristic has
  `batch_count`; CG has no batching concept but does have
  `only_improving_columns` / pricing-iteration controls). Rather than
  forcing them into one shared grid (which would either drop
  Heuristic's batch_count sweep or run CG redundantly once per
  batch_count with no effect), each algorithm is swept over its own
  full grid, independently, for every route -- exactly the sweep each
  standalone script already ran -- and the results are combined only at
  the reporting layer (the three CSVs above), not by forcing a shared
  parameter space that doesn't actually exist between the two
  algorithms.
* Visualization defaults to ONE Amazon-vs-Heuristic figure and ONE
  Amazon-vs-CG figure per route (the best result from each algorithm's
  grid), not one figure per grid combination -- with two full grids run
  for every one of potentially 1000 routes, plotting every combination
  would produce an unmanageable number of files. Pass --plot-all-combos
  to instead render every grid combination for both algorithms
  (Heuristic's and CG's grids independently -- there is no longer a
  cross-product between them now that each is plotted against Amazon
  separately); this is almost certainly not what you want above a
  handful of routes.
"""

# --- CRITICAL CPU & THERMAL LIMITS (matches both standalone scripts) ---
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import gc
import itertools
import time
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from algo_data_loader import AmazonDataLoader, compute_open_route_cost
from algo_hybrid_LRWSQAOA import run_algo_hybrid_2_5
from cg_hybrid_lrwsqaoa_sub import run_cg_hybrid_lrwsqaoa_sub, ITERATION_CG
from plot_publication import generate_overall_visualizations


# ---------------------------------------------------------------------
# Route sampling -- defined once here (see module docstring: this was
# previously duplicated verbatim in run_amazon_experiment.py AND
# run_CG_experiment.py).
# ---------------------------------------------------------------------

def get_amazon_dataset_sample(data_dir="./almrrc2021-data-training", num_routes=10, seed=2026):
    """Loads real ALMRRC routes. Identical logic/seed handling to both
    standalone scripts, so pointing any of the three scripts at the same
    (data_dir, num_routes, seed) draws the same route sample."""
    if not os.path.exists(data_dir) and os.path.exists("./data"):
        data_dir = "./data"

    loader = AmazonDataLoader(data_dir=data_dir)

    if not loader.travel_times:
        raise FileNotFoundError(
            f"No route data found in '{data_dir}'. Ensure travel_times.json is available."
        )

    all_route_ids = sorted(list(loader.travel_times.keys()))

    if num_routes is not None and num_routes < len(all_route_ids):
        rng = np.random.default_rng(seed)
        selected_route_ids = rng.choice(all_route_ids, size=num_routes, replace=False).tolist()
    else:
        selected_route_ids = all_route_ids

    print(f"--> Selected {len(selected_route_ids)} route(s) out of {len(all_route_ids)} total routes.")

    dataset = []
    for rid in selected_route_ids:
        extracted = loader.extract_single_route(rid)
        matrix = np.array(extracted["matrix"])
        coords = np.array(extracted["coords"])

        if coords is None or np.all(coords == 0):
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed)
            coords = mds.fit_transform(matrix)

        dataset.append({
            "route_id": extracted.get("route_id", rid),
            "n_nodes": extracted["n_nodes"],
            "coords": coords,
            "matrix": matrix,
            "depot_idx": extracted.get("depot_idx", 0),
            "amazon_planned_tour": extracted["amazon_planned_sequence"],
        })

    return dataset


# ---------------------------------------------------------------------
# Per-route, per-algorithm grid execution
# ---------------------------------------------------------------------

def _run_heuristic_grid_for_route(data, amazon_cost, param_grid, seed, keep_all_tours=False):
    """Runs Algorithm 1 over its full parameter grid for one route.
    Returns (records, best_record) where best_record is the row with
    the lowest heuristic_cost (None if the grid is empty).

    Tours are memory-expensive to retain across a full 1000-route x
    full-grid run, so by default only the best record's tour is kept
    (best["_tour"]) -- enough for the default best-vs-best 3-way plot.
    Pass keep_all_tours=True (wired up automatically when
    --plot-all-combos is set) to retain every record's tour instead,
    which --plot-all-combos needs to render every combo pairing."""
    matrix = data["matrix"]
    records = []
    best = None

    for q, exp, b, xy in param_grid:
        param_str = f"h_q{q}_exp{int(exp*100)}_b{b}_xy{1 if xy else 0}"
        t0 = time.time()
        res = run_algo_hybrid_2_5(
            data, qubit_count=q, exploration_percent=exp,
            batch_count=b, xy_mixer=xy, seed=seed,
        )
        elapsed = time.time() - t0

        tour = res["tour"]
        cost = compute_open_route_cost(tour, matrix)
        cost_diff_abs = cost - amazon_cost
        cost_diff_pct = (cost_diff_abs / amazon_cost) * 100.0 if amazon_cost else 0.0

        rec = {
            "route_id": data["route_id"], "n_nodes": data["n_nodes"], "algorithm": "heuristic",
            "qubit_count": q, "exploration_percent": exp, "batch_count": b, "xy_mixer": xy,
            "only_improving_columns": None,
            "amazon_cost": round(amazon_cost, 2), "algo_cost": round(cost, 2),
            "cost_diff_abs": round(cost_diff_abs, 2), "cost_diff_pct": round(cost_diff_pct, 2),
            "improvement_pct": round(-cost_diff_pct, 2), "runtime_sec": round(elapsed, 3),
            "num_iterations_run": None, "num_pool_columns_final": None,
            "num_segments_selected": None, "final_master_status": None, "pre_2opt_cost": None,
            "param_str": param_str, "error": None,
        }
        if keep_all_tours:
            rec["_tour"] = tour
        records.append(rec)
        if best is None or cost < best["algo_cost"]:
            best = dict(rec)
            best["_tour"] = tour

    return records, best


def _run_cg_grid_for_route(data, amazon_cost, param_grid, max_pricing_nodes, n_iterations, time_limit, seed,
                            keep_all_tours=False):
    """Runs Algorithm 2 over its full parameter grid for one route.
    Returns (records, best_record); failures are recorded but excluded
    from best-selection, matching run_CG_experiment.py's error handling.
    See _run_heuristic_grid_for_route's docstring re: keep_all_tours."""
    matrix = data["matrix"]
    records = []
    best = None

    for q, exp, xy, only_improving in param_grid:
        param_str = f"cg_q{q}_exp{int(exp*100)}_xy{1 if xy else 0}_imp{1 if only_improving else 0}"
        t0 = time.time()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = run_cg_hybrid_lrwsqaoa_sub(
                    data, qubit_count=q, exploration_percent=exp, xy_mixer=xy,
                    only_improving_columns=only_improving, max_pricing_nodes=max_pricing_nodes,
                    n_iterations=n_iterations, time_limit=time_limit, seed=seed,
                )
                for w in caught:
                    print(f"    [warning] {w.message}")
        except Exception as e:
            elapsed = time.time() - t0
            records.append({
                "route_id": data["route_id"], "n_nodes": data["n_nodes"], "algorithm": "cg",
                "qubit_count": q, "exploration_percent": exp, "batch_count": None, "xy_mixer": xy,
                "only_improving_columns": only_improving,
                "amazon_cost": round(amazon_cost, 2), "algo_cost": None,
                "cost_diff_abs": None, "cost_diff_pct": None, "improvement_pct": None,
                "runtime_sec": round(elapsed, 3),
                "num_iterations_run": None, "num_pool_columns_final": None,
                "num_segments_selected": None, "final_master_status": None, "pre_2opt_cost": None,
                "param_str": param_str, "error": str(e),
            })
            continue
        elapsed = time.time() - t0

        tour = res["tour"]
        cost = compute_open_route_cost(tour, matrix)
        cost_diff_abs = cost - amazon_cost
        cost_diff_pct = (cost_diff_abs / amazon_cost) * 100.0 if amazon_cost else 0.0
        diag = res["cg_diagnostics"]

        rec = {
            "route_id": data["route_id"], "n_nodes": data["n_nodes"], "algorithm": "cg",
            "qubit_count": q, "exploration_percent": exp, "batch_count": None, "xy_mixer": xy,
            "only_improving_columns": only_improving,
            "amazon_cost": round(amazon_cost, 2), "algo_cost": round(cost, 2),
            "cost_diff_abs": round(cost_diff_abs, 2), "cost_diff_pct": round(cost_diff_pct, 2),
            "improvement_pct": round(-cost_diff_pct, 2), "runtime_sec": round(elapsed, 3),
            "num_iterations_run": diag["num_iterations_run"],
            "num_pool_columns_final": diag["num_pool_columns_final"],
            "num_segments_selected": diag["num_segments_selected"],
            "final_master_status": diag["final_master_status"],
            "pre_2opt_cost": round(diag["pre_2opt_cost"], 2),
            "param_str": param_str, "error": None,
        }
        if keep_all_tours:
            rec["_tour"] = tour
        records.append(rec)
        if best is None or cost < best["algo_cost"]:
            best = dict(rec)
            best["_tour"] = tour

    return records, best


# ---------------------------------------------------------------------
# Main grid driver
# ---------------------------------------------------------------------

def run_all_experiment_grid(
    data_dir="./almrrc2021-data-training",
    num_routes=10,
    output_dir="./experiment_results_all",
    seed=2026,
    # --- Algorithm 1 (Heuristic) grid ---
    heuristic_qubit_counts=(2, 3),
    heuristic_exploration_percents=(0.0, 0.2),
    heuristic_batch_counts=(1, 2, 3, 4),
    heuristic_xy_mixers=(False,),
    # --- Algorithm 2 (CG / TDE-QP) grid ---
    cg_qubit_counts=(2, 3),
    cg_exploration_percents=(0.0, 0.2),
    cg_xy_mixers=(False,),
    cg_only_improving_columns_options=(True,),
    cg_max_pricing_nodes=None,
    cg_n_iterations=ITERATION_CG,
    cg_time_limit=60,
    # --- Visualization ---
    generate_plots=True,
    plot_all_combos=False,
    plot_max_routes=None,
):
    """Runs both algorithms' full grids, for every route in the sample,
    and writes the three combined CSVs + the 3-way comparison figures
    described in the module docstring."""
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualise_experiments_ALL")
    heuristic_viz_dir = os.path.join(viz_dir, "heuristic_vs_amazon")
    cg_viz_dir = os.path.join(viz_dir, "cg_vs_amazon")

    routes_data = get_amazon_dataset_sample(data_dir=data_dir, num_routes=num_routes, seed=seed)

    heuristic_grid = [
        (q, exp, b, xy)
        for q, exp, b, xy in itertools.product(
            heuristic_qubit_counts, heuristic_exploration_percents,
            heuristic_batch_counts, heuristic_xy_mixers,
        )
        if b <= q
    ]
    cg_grid = list(itertools.product(
        cg_qubit_counts, cg_exploration_percents, cg_xy_mixers, cg_only_improving_columns_options,
    ))

    print(
        f"\n=== Starting UNIFIED Grid Search Benchmark (Heuristic + CG) ==="
        f"\n  Routes Sampled          : {len(routes_data)}"
        f"\n  Heuristic Grid Combos   : {len(heuristic_grid)}"
        f"\n  CG Grid Combos          : {len(cg_grid)}"
        f"\n  CG Pricing Iterations   : {cg_n_iterations} (cap; may converge earlier)"
        f"\n  Total Algorithm Runs    : {len(routes_data) * (len(heuristic_grid) + len(cg_grid))}\n"
    )

    all_records = []
    comparison_rows = []
    t_start_grid = time.time()

    for r_idx, data in enumerate(routes_data, 1):
        route_id = data["route_id"]
        matrix = data["matrix"]
        amazon_tour = data["amazon_planned_tour"]
        amazon_cost = compute_open_route_cost(amazon_tour, matrix)

        t_route_start = time.time()

        will_plot_this_route = generate_plots and (plot_max_routes is None or r_idx <= plot_max_routes)
        keep_all = will_plot_this_route and plot_all_combos

        h_records, h_best = _run_heuristic_grid_for_route(
            data, amazon_cost, heuristic_grid, seed, keep_all_tours=keep_all,
        )
        c_records, c_best = _run_cg_grid_for_route(
            data, amazon_cost, cg_grid, cg_max_pricing_nodes, cg_n_iterations, cg_time_limit, seed,
            keep_all_tours=keep_all,
        )
        all_records.extend(h_records)
        all_records.extend(c_records)

        h_cost_str = f"{h_best['algo_cost']:.2f}" if h_best else "N/A"
        c_cost_str = f"{c_best['algo_cost']:.2f}" if c_best else "N/A (all combos failed)"
        print(
            f"[{r_idx}/{len(routes_data)}] Route {route_id} | Amazon: {amazon_cost:.2f} | "
            f"Best Heuristic: {h_cost_str} ({h_best['param_str'] if h_best else '-'}) | "
            f"Best CG: {c_cost_str} ({c_best['param_str'] if c_best else '-'}) | "
            f"Route time: {time.time() - t_route_start:.2f}s"
        )

        comparison_rows.append({
            "route_id": route_id,
            "n_nodes": data["n_nodes"],
            "amazon_cost": round(amazon_cost, 2),
            "best_heuristic_cost": h_best["algo_cost"] if h_best else None,
            "best_heuristic_params": h_best["param_str"] if h_best else None,
            "best_heuristic_runtime_sec": h_best["runtime_sec"] if h_best else None,
            "heuristic_improvement_pct_vs_amazon": h_best["improvement_pct"] if h_best else None,
            "best_cg_cost": c_best["algo_cost"] if c_best else None,
            "best_cg_params": c_best["param_str"] if c_best else None,
            "best_cg_runtime_sec": c_best["runtime_sec"] if c_best else None,
            "cg_improvement_pct_vs_amazon": c_best["improvement_pct"] if c_best else None,
            "heuristic_vs_cg_pct": (
                round(-100.0 * (h_best["algo_cost"] - c_best["algo_cost"]) / c_best["algo_cost"], 2)
                if (h_best and c_best and c_best["algo_cost"]) else None
            ),
            "winner": (
                "Amazon" if (not h_best and not c_best) else
                "Heuristic" if (h_best and (not c_best or h_best["algo_cost"] <= c_best["algo_cost"]) and h_best["algo_cost"] < amazon_cost) else
                "CG" if (c_best and c_best["algo_cost"] < amazon_cost) else
                "Amazon"
            ),
        })

        if will_plot_this_route:
            if plot_all_combos:
                # keep_all_tours=True above guarantees every h_records /
                # c_records entry carries its own "_tour". Each
                # algorithm is plotted against Amazon independently --
                # there is no cross-product between the two grids to
                # iterate, since each figure is a 2-panel Amazon-vs-one-
                # algorithm comparison, not a combined 3-panel figure.
                for hr in h_records:
                    generate_overall_visualizations(
                        data, hr["_tour"], hr["algo_cost"], hr["param_str"], heuristic_viz_dir,
                        algo_label="Heuristic (WS-LR-QAOA)", algo_color="#1a6b1a",
                    )
                for cr in c_records:
                    if cr.get("error"):
                        continue
                    generate_overall_visualizations(
                        data, cr["_tour"], cr["algo_cost"], cr["param_str"], cg_viz_dir,
                        algo_label="CG (TDE-QP)", algo_color="#6a3d9a",
                    )
            else:
                if h_best:
                    generate_overall_visualizations(
                        data, h_best["_tour"], h_best["algo_cost"], h_best["param_str"], heuristic_viz_dir,
                        algo_label="Heuristic (WS-LR-QAOA)", algo_color="#1a6b1a",
                    )
                if c_best:
                    generate_overall_visualizations(
                        data, c_best["_tour"], c_best["algo_cost"], c_best["param_str"], cg_viz_dir,
                        algo_label="CG (TDE-QP)", algo_color="#6a3d9a",
                    )

        gc.collect()

    # ------------------ Write combined CSVs ------------------
    df_all = pd.DataFrame(all_records).drop(columns=["_tour"], errors="ignore")
    detailed_csv = os.path.join(output_dir, "experiment_ALL_detailed_results.csv")
    df_all.to_csv(detailed_csv, index=False)

    df_h_valid = df_all[(df_all["algorithm"] == "heuristic") & df_all["error"].isna()]
    df_c_valid = df_all[(df_all["algorithm"] == "cg") & df_all["error"].isna()]

    summary_frames = []
    if len(df_h_valid) > 0:
        summary_frames.append(
            df_h_valid.groupby(["algorithm", "qubit_count", "exploration_percent", "batch_count", "xy_mixer"])
            .agg(
                mean_amazon_cost=("amazon_cost", "mean"),
                mean_algo_cost=("algo_cost", "mean"),
                mean_improvement_pct=("improvement_pct", "mean"),
                win_rate_pct=("improvement_pct", lambda x: (x > 0).mean() * 100),
                mean_runtime_sec=("runtime_sec", "mean"),
            ).reset_index()
        )
    if len(df_c_valid) > 0:
        summary_frames.append(
            df_c_valid.groupby(["algorithm", "qubit_count", "exploration_percent", "xy_mixer", "only_improving_columns"])
            .agg(
                mean_amazon_cost=("amazon_cost", "mean"),
                mean_algo_cost=("algo_cost", "mean"),
                mean_improvement_pct=("improvement_pct", "mean"),
                win_rate_pct=("improvement_pct", lambda x: (x > 0).mean() * 100),
                mean_segments_selected=("num_segments_selected", "mean"),
                mean_iterations_run=("num_iterations_run", "mean"),
                mean_runtime_sec=("runtime_sec", "mean"),
            ).reset_index()
        )
    df_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_csv = os.path.join(output_dir, "experiment_ALL_summary_by_parameters.csv")
    df_summary.to_csv(summary_csv, index=False)

    df_comparison = pd.DataFrame(comparison_rows)
    comparison_csv = os.path.join(output_dir, "experiment_ALL_route_comparison.csv")
    df_comparison.to_csv(comparison_csv, index=False)

    print("\n" + "=" * 100)
    print("                              UNIFIED BENCHMARK SUMMARY (by parameters)                          ")
    print("=" * 100)
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    print("=" * 100)
    print("\n" + "=" * 100)
    print("                        PER-ROUTE COMPARISON (best config per algorithm)                        ")
    print("=" * 100)
    print(df_comparison.to_string(index=False))
    print("=" * 100)
    if not df_comparison.empty:
        win_counts = df_comparison["winner"].value_counts()
        print(f"\nWinner counts across {len(df_comparison)} routes:\n{win_counts.to_string()}")
    print(f"\n--> Detailed results   : {detailed_csv}")
    print(f"--> Parameter summary  : {summary_csv}")
    print(f"--> Route comparison   : {comparison_csv}")
    print(f"--> Visualizations     : {viz_dir}")
    print(f"--> Total Execution Time: {time.time() - t_start_grid:.2f}s\n")

    return df_all, df_summary, df_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BOTH the Heuristic (Algorithm 1) and CG/TDE-QP (Algorithm 2) grids "
                    "against the Amazon baseline, on the same route sample, in one pass."
    )
    # --- Shared / dataset parameters ---
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training", help="Dataset directory")
    parser.add_argument("--num-routes", type=int, default=1, help="Number of routes to sample")
    parser.add_argument("--output-dir", type=str, default="./experiment_results_all", help="Output directory for CSVs & plots")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (shared route sample + algorithm seeding)")
    parser.add_argument("--no-plots", action="store_true", help="Disable generating visual plots entirely")
    parser.add_argument("--plot-all-combos", action="store_true",
                        help="Plot every (heuristic-combo x cg-combo) pairing per route instead of only the "
                             "best-vs-best comparison. Produces MANY files above a handful of routes.")
    parser.add_argument("--plot-max-routes", type=int, default=None,
                        help="Only generate plots for the first N routes in the sample (all routes still get "
                             "full CSV results). Useful to cap plot volume on large --num-routes runs.")

    # --- Algorithm 1 (Heuristic) grid ---
    parser.add_argument("--h-qubit-counts", type=int, nargs="+", default=[2, 3],
                        help="Heuristic qubit budgets to sweep (k). Simulation cost scales as 2**(k**2); "
                             "k=4 is disabled by default (see algorithms paper, Sec. Discussion).")
    parser.add_argument("--h-exploration-percents", type=float, nargs="+", default=[0.0, 0.2],
                        help="Heuristic exploration fractions rho to sweep.")
    parser.add_argument("--h-batch-counts", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Heuristic batch commitment sizes b to sweep (filtered to b <= k per combo).")
    parser.add_argument("--h-xy-mixer", action="store_true",
                        help="Also sweep xy_mixer=True for the Heuristic grid (default: WS mixer only).")

    # --- Algorithm 2 (CG / TDE-QP) grid ---
    parser.add_argument("--cg-qubit-counts", type=int, nargs="+", default=[2, 3],
                        help="CG pricing qubit budgets to sweep (k).")
    parser.add_argument("--cg-exploration-percents", type=float, nargs="+", default=[0.0],
                        help="CG exploration fractions rho to sweep.")
    parser.add_argument("--cg-xy-mixer", action="store_true",
                        help="Also sweep xy_mixer=True for the CG grid (default: WS mixer only).")
    parser.add_argument("--cg-allow-non-improving-columns", action="store_true",
                        help="Also sweep only_improving_columns=False (default: True only).")
    parser.add_argument("--cg-max-pricing-nodes", type=int, default=None,
                        help="Subsample this many starting nodes for CG pricing instead of every node "
                             "(speed/quality tradeoff; default: every node, i.e. O(n) per iteration).")
    parser.add_argument("--cg-iterations", type=int, default=ITERATION_CG,
                        help=f"CG pricing iterations cap per run (default: ITERATION_CG={ITERATION_CG}).")
    parser.add_argument("--cg-time-limit", type=int, default=60,
                        help="CBC master-problem solver time limit (seconds) per solve.")

    args = parser.parse_args()

    run_all_experiment_grid(
        data_dir=args.data_dir,
        num_routes=args.num_routes,
        output_dir=args.output_dir,
        seed=args.seed,
        heuristic_qubit_counts=tuple(args.h_qubit_counts),
        heuristic_exploration_percents=tuple(args.h_exploration_percents),
        heuristic_batch_counts=tuple(args.h_batch_counts),
        heuristic_xy_mixers=(False, True) if args.h_xy_mixer else (False,),
        cg_qubit_counts=tuple(args.cg_qubit_counts),
        cg_exploration_percents=tuple(args.cg_exploration_percents),
        cg_xy_mixers=(False, True) if args.cg_xy_mixer else (False,),
        cg_only_improving_columns_options=(True, False) if args.cg_allow_non_improving_columns else (True,),
        cg_max_pricing_nodes=args.cg_max_pricing_nodes,
        cg_n_iterations=args.cg_iterations,
        cg_time_limit=args.cg_time_limit,
        generate_plots=not args.no_plots,
        plot_all_combos=args.plot_all_combos,
        plot_max_routes=args.plot_max_routes,
    )
