"""
run_CG_experiment.py

Grid search experimental runner for the Column-Generation Hybrid algorithm
(cg_hybrid_lrwsqaoa_sub.py) evaluated against Amazon Planned routes on the
ALMRRC dataset. Mirrors run_amazon_experiment.py's structure (same data
loading, same CSV conventions) so results from the two experiments are
directly comparable; also reuses plot_publication.py for figures instead
of duplicating plotting code.

Grid parameters (no batch_count here -- the CG algorithm has no
receding-horizon batching concept; coverage is decided by the master
problem in one shot):
  - qubit_count: [2, 3, 4]
  - exploration_percent: [0.0, 0.2]
  - xy_mixer: [False]  (True left available but off by default, matching
    run_amazon_experiment.py's current default)
  - only_improving_columns: [True]
"""

# --- CRITICAL CPU & THERMAL LIMITS (matches run_amazon_experiment.py) ---
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
from cg_hybrid_lrwsqaoa_sub import run_cg_hybrid_lrwsqaoa_sub
from plot_publication import generate_overall_visualizations


def get_amazon_dataset_sample(data_dir="./almrrc2021-data-training", num_routes=10, seed=2026):
    """Identical sampling logic to run_amazon_experiment.py, so both
    experiments can be pointed at the same route sample via the same seed."""
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


def run_cg_experiment_grid(
    data_dir="./almrrc2021-data-training",
    num_routes=10,
    qubit_counts=(2, 3, 4),
    exploration_percents=(0.0, 0.2),
    xy_mixers=(False,),
    only_improving_columns_options=(True,),
    max_pricing_nodes=None,
    time_limit=60,
    output_dir="./experiment_results_cg",
    seed=2026,
    generate_plots=True,
):
    """Executes the CG algorithm's grid search benchmark against Amazon
    Planned routes, on the same route sample run_amazon_experiment.py
    would draw for the same (data_dir, num_routes, seed)."""
    os.makedirs(output_dir, exist_ok=True)

    routes_data = get_amazon_dataset_sample(data_dir=data_dir, num_routes=num_routes, seed=seed)

    param_grid = list(itertools.product(
        qubit_counts, exploration_percents, xy_mixers, only_improving_columns_options
    ))

    total_runs = len(routes_data) * len(param_grid)
    print(
        f"\n=== Starting CG Grid Search Benchmark ==="
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

        for q, exp, xy, only_improving in param_grid:
            run_counter += 1
            param_str = (
                f"cg_q{q}_exp{int(exp*100)}_xy{1 if xy else 0}"
                f"_imp{1 if only_improving else 0}"
            )

            t0 = time.time()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res = run_cg_hybrid_lrwsqaoa_sub(
                        data,
                        qubit_count=q,
                        exploration_percent=exp,
                        xy_mixer=xy,
                        only_improving_columns=only_improving,
                        max_pricing_nodes=max_pricing_nodes,
                        time_limit=time_limit,
                        seed=seed,
                    )
                    for w in caught:
                        print(f"    [warning] {w.message}")
            except Exception as e:
                elapsed = time.time() - t0
                print(
                    f"[{run_counter}/{total_runs}] Route {route_id} ({r_idx}/{len(routes_data)}) | "
                    f"Params: {param_str} | FAILED after {elapsed:.2f}s: {e}"
                )
                records.append({
                    "route_id": route_id,
                    "n_nodes": data["n_nodes"],
                    "qubit_count": q,
                    "exploration_percent": exp,
                    "xy_mixer": xy,
                    "only_improving_columns": only_improving,
                    "amazon_cost": round(amazon_cost, 2),
                    "cg_cost": None,
                    "cost_diff_abs": None,
                    "cost_diff_pct": None,
                    "improvement_pct": None,
                    "runtime_sec": round(elapsed, 3),
                    "error": str(e),
                })
                continue
            elapsed = time.time() - t0

            cg_tour = res["tour"]
            cg_cost = compute_open_route_cost(cg_tour, matrix)
            cost_diff_abs = cg_cost - amazon_cost
            cost_diff_pct = (cost_diff_abs / amazon_cost) * 100.0 if amazon_cost else 0.0
            improvement_pct = -cost_diff_pct

            diag = res["cg_diagnostics"]
            records.append({
                "route_id": route_id,
                "n_nodes": data["n_nodes"],
                "qubit_count": q,
                "exploration_percent": exp,
                "xy_mixer": xy,
                "only_improving_columns": only_improving,
                "amazon_cost": round(amazon_cost, 2),
                "cg_cost": round(cg_cost, 2),
                "cost_diff_abs": round(cost_diff_abs, 2),
                "cost_diff_pct": round(cost_diff_pct, 2),
                "improvement_pct": round(improvement_pct, 2),
                "runtime_sec": round(elapsed, 3),
                "num_priced_columns": diag["num_priced_columns"],
                "num_pool_columns": diag["num_pool_columns_after_dedupe"],
                "num_segments_selected": diag["num_segments_selected"],
                "round1_lp_status": diag["round1_lp_status"],
                "round2_master_status": diag["round2_master_status"],
                "pre_2opt_cost": round(diag["pre_2opt_cost"], 2),
                "error": None,
            })

            print(
                f"[{run_counter}/{total_runs}] Route {route_id} ({r_idx}/{len(routes_data)}) | "
                f"Params: {param_str} | Amazon: {amazon_cost:.2f} | CG: {cg_cost:.2f} | "
                f"Improv: {improvement_pct:+.2f}% | Segments: {diag['num_segments_selected']} | "
                f"Time: {elapsed:.2f}s"
            )

            if generate_plots:
                generate_overall_visualizations(data, cg_tour, cg_cost, param_str, output_dir)

        gc.collect()

    df_results = pd.DataFrame(records)

    csv_path = os.path.join(output_dir, "cg_experiment_detailed_results.csv")
    df_results.to_csv(csv_path, index=False)

    valid = df_results[df_results["error"].isna()]
    if len(valid) > 0:
        df_summary = (
            valid.groupby(["qubit_count", "exploration_percent", "xy_mixer", "only_improving_columns"])
            .agg(
                mean_amazon_cost=("amazon_cost", "mean"),
                mean_cg_cost=("cg_cost", "mean"),
                mean_improvement_pct=("improvement_pct", "mean"),
                win_rate_pct=("improvement_pct", lambda x: (x > 0).mean() * 100),
                mean_segments_selected=("num_segments_selected", "mean"),
                mean_runtime_sec=("runtime_sec", "mean"),
            )
            .reset_index()
            .sort_values(by="mean_improvement_pct", ascending=False)
        )
        summary_csv_path = os.path.join(output_dir, "cg_experiment_summary_by_parameters.csv")
        df_summary.to_csv(summary_csv_path, index=False)

        print("\n" + "=" * 90)
        print("                   CG EXPERIMENT BENCHMARK SUMMARY TABLE                        ")
        print("=" * 90)
        print(df_summary.to_string(index=False))
        print("=" * 90)
        print(f"\n--> Full detailed results saved to: {csv_path}")
        print(f"--> Parameter summary table saved to: {summary_csv_path}")
    else:
        df_summary = pd.DataFrame()
        print("\nAll runs failed -- check the 'error' column in the detailed CSV.")

    print(f"--> Total Execution Time: {time.time() - t_start_grid:.2f}s\n")

    return df_results, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CG Hybrid LRWSQAOA-Sub Parameter Grid Search vs Amazon Baseline"
    )
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training", help="Dataset directory")
    parser.add_argument("--num-routes", type=int, default=1, help="Number of routes to sample")
    parser.add_argument("--output-dir", type=str, default="./experiment_results_cg", help="Output directory for CSVs & plots")
    parser.add_argument("--no-plots", action="store_true", help="Disable generating visual plots to speed up runs")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument(
        "--max-pricing-nodes", type=int, default=None,
        help="Subsample this many starting nodes for pricing instead of every node (speed/quality tradeoff)",
    )
    parser.add_argument("--time-limit", type=int, default=60, help="CBC solver time limit (seconds) per master solve")

    args = parser.parse_args()

    run_cg_experiment_grid(
        data_dir=args.data_dir,
        num_routes=args.num_routes,
        qubit_counts=(2, 3, 4),
        exploration_percents=(0.0, 0.2),
        xy_mixers=(False,),
        only_improving_columns_options=(True,),
        max_pricing_nodes=args.max_pricing_nodes,
        time_limit=args.time_limit,
        output_dir=args.output_dir,
        seed=args.seed,
        generate_plots=not args.no_plots,
    )
