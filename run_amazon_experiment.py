"""
run_amazon_experiment.py

Grid search experimental runner for Hybrid Algorithm (WS-LR QAOA + LNS)
evaluated against Amazon Planned routes on the ALMRRC dataset.

Grid Parameters:
  - qubit_count: [2, 3, 4]
  - exploration_percent: [0.0, 0.2]
  - batch_count: [1, 2, 3, 4] (filtered so batch_count <= qubit_count)
  - xy_mixer: [False, True]

--------------------------------------------------------------------------
FIX LOG (this revision)
--------------------------------------------------------------------------
* Visualization: this file used to define its own local
  generate_overall_visualizations() (plain matplotlib, per-node index
  labels on every stop, no arrows/scorecard/vector export). That is now
  removed. Both experiment runners import the SAME publication-quality
  plotting function from plot_publication.py, exactly like
  run_CG_experiment.py already did -- so Hybrid-vs-Amazon and
  CG-vs-Amazon figures are visually consistent (same fonts, same
  directional arrows, same scorecard banner, same PNG+PDF export) and
  the plotting code can no longer silently drift between the two
  experiments. algo_label/algo_color are passed explicitly so the panel
  is correctly labeled "Hybrid Algo 2+5" (CG's runner passes its own
  label/color instead of relying on the old default).
* __main__ block's actual arguments (qubit_counts=[2, 3], xy_mixers=
  [False]) previously did not match this docstring's stated grid
  ([2, 3, 4] and [False, True]) or the run_experiment_grid() defaults.
  The docstring/defaults above and the __main__ call below are now
  kept in sync; see the note next to __main__ if you re-enable q=4
  or xy_mixer=True.
--------------------------------------------------------------------------
"""

# --- CRITICAL CPU & THERMAL LIMITS ---
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
import numpy as np
import pandas as pd

# Non-interactive backend to prevent GUI blocking
import matplotlib
matplotlib.use("Agg")

from algo_data_loader import AmazonDataLoader, compute_open_route_cost
from algo_hybrid_LRWSQAOA import run_algo_hybrid_2_5
from plot_publication import generate_overall_visualizations


def get_amazon_dataset_sample(data_dir="./almrrc2021-data-training", num_routes=10, seed=2026):
    """
    Loads real Amazon Last Mile Routing Challenge routes using AmazonDataLoader.
    Supports limiting the number of sampled routes via `num_routes`.
    """
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


def run_experiment_grid(
    data_dir="./almrrc2021-data-training",
    num_routes=10,
    qubit_counts=[2, 3, 4],
    exploration_percents=[0.0, 0.2],
    batch_counts=[1, 2, 3, 4],
    xy_mixers=[False, True],
    output_dir="./experiment_results",
    seed=2026,
    generate_plots=True,
):
    """Executes full grid search benchmark against Amazon Planned routes."""
    os.makedirs(output_dir, exist_ok=True)

    routes_data = get_amazon_dataset_sample(data_dir=data_dir, num_routes=num_routes, seed=seed)

    # Valid parameter combinations enforcing batch_count <= qubit_count
    param_grid = [
        (q, exp, b, xy)
        for q, exp, b, xy in itertools.product(qubit_counts, exploration_percents, batch_counts, xy_mixers)
        if b <= q
    ]

    total_runs = len(routes_data) * len(param_grid)
    print(
        f"\n=== Starting Grid Search Benchmark ==="
        f"\n  Routes Sampled     : {len(routes_data)}"
        f"\n  Valid Grid Combos  : {len(param_grid)}"
        f"\n  Total Iterations   : {total_runs}\n"
    )

    records = []
    run_counter = 0
    t_start_grid = time.time()

    for r_idx, data in enumerate(routes_data, 1):
        route_id = data["route_id"]
        matrix = data["matrix"]
        amazon_tour = data["amazon_planned_tour"]
        amazon_cost = compute_open_route_cost(amazon_tour, matrix)

        for q, exp, b, xy in param_grid:
            run_counter += 1
            param_str = f"q{q}_exp{int(exp*100)}_b{b}_xy{1 if xy else 0}"

            t0 = time.time()
            res = run_algo_hybrid_2_5(
                data,
                qubit_count=q,
                exploration_percent=exp,
                batch_count=b,
                xy_mixer=xy,
                seed=seed,
            )
            elapsed = time.time() - t0

            hybrid_tour = res["tour"]
            hybrid_cost = compute_open_route_cost(hybrid_tour, matrix)
            cost_diff_abs = hybrid_cost - amazon_cost
            cost_diff_pct = (cost_diff_abs / amazon_cost) * 100.0 if amazon_cost else 0.0
            improvement_pct = -cost_diff_pct

            records.append({
                "route_id": route_id,
                "n_nodes": data["n_nodes"],
                "qubit_count": q,
                "exploration_percent": exp,
                "batch_count": b,
                "xy_mixer": xy,
                "amazon_cost": round(amazon_cost, 2),
                "hybrid_cost": round(hybrid_cost, 2),
                "cost_diff_abs": round(cost_diff_abs, 2),
                "cost_diff_pct": round(cost_diff_pct, 2),
                "improvement_pct": round(improvement_pct, 2),
                "runtime_sec": round(elapsed, 3),
            })

            print(
                f"[{run_counter}/{total_runs}] Route {route_id} ({r_idx}/{len(routes_data)}) | "
                f"Params: {param_str} | Amazon: {amazon_cost:.2f} | Hybrid: {hybrid_cost:.2f} | "
                f"Improv: {improvement_pct:+.2f}% | Time: {elapsed:.2f}s"
            )

            if generate_plots:
                # Shared publication-quality plotting (same module + look as
                # run_CG_experiment.py). algo_label/algo_color are passed
                # explicitly so the right panel is always correctly labeled,
                # even though these two happen to match the function's
                # defaults today.
                generate_overall_visualizations(
                    data, hybrid_tour, hybrid_cost, param_str, output_dir,
                    algo_label="Hybrid Algo 2+5 (WS-LR-QAOA + LNS)",
                    algo_color="#1a6b1a",
                )

        gc.collect()

    df_results = pd.DataFrame(records)

    # Save detailed CSV
    csv_path = os.path.join(output_dir, "experiment_detailed_results.csv")
    df_results.to_csv(csv_path, index=False)

    # Grouped Summary across parameter configurations
    df_summary = (
        df_results.groupby(["qubit_count", "exploration_percent", "batch_count", "xy_mixer"])
        .agg(
            mean_amazon_cost=("amazon_cost", "mean"),
            mean_hybrid_cost=("hybrid_cost", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            win_rate_pct=("improvement_pct", lambda x: (x > 0).mean() * 100),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
        .sort_values(by="mean_improvement_pct", ascending=False)
    )

    summary_csv_path = os.path.join(output_dir, "experiment_summary_by_parameters.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 90)
    print("                      EXPERIMENTAL BENCHMARK SUMMARY TABLE                       ")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)
    print(f"\n--> Full detailed results saved to: {csv_path}")
    print(f"--> Parameter summary table saved to: {summary_csv_path}")
    print(f"--> Total Execution Time: {time.time() - t_start_grid:.2f}s\n")

    return df_results, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid QAOA Parameter Grid Search vs Amazon Baseline")
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training", help="Dataset directory")
    parser.add_argument("--num-routes", type=int, default=1000, help="Number of routes to sample (default: 10)")
    parser.add_argument("--output-dir", type=str, default="./experiment_results", help="Output directory for CSVs & plots")
    parser.add_argument("--no-plots", action="store_true", help="Disable generating visual plots to speed up runs")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")

    args = parser.parse_args()

    # NOTE: qubit_count=4 and xy_mixer=True are left disabled below (see
    # PERFORMANCE NOTE in algo_hybrid_LRWSQAOA.py -- statevector size is
    # 2**(qubit_count**2), so q=4 means simulating 65,536 amplitudes per
    # COBYLA step, per sub-tour, per route). Re-enable by editing the lists
    # below once you've budgeted for the runtime; this now matches the
    # module docstring above instead of silently diverging from it.
    run_experiment_grid(
        data_dir=args.data_dir,
        num_routes=args.num_routes,
        qubit_counts=[2, 3],
        exploration_percents=[0.0, 0.2],
        batch_counts=[1, 2, 3, 4],
        xy_mixers=[False],
        output_dir=args.output_dir,
        seed=args.seed,
        generate_plots=not args.no_plots,
    )
