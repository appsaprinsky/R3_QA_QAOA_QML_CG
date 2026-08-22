"""
run_amazon_experiment.py

Grid search experimental runner for Hybrid Algorithm 2+5 (WS-LR QAOA + LNS)
evaluated against Amazon Planned routes on the ALMRRC dataset.

Grid Parameters:
  - qubit_count: [2, 3, 4]
  - exploration_percent: [0.0, 0.2]
  - batch_count: [1, 2, 3, 4] (filtered so batch_count <= qubit_count)
  - xy_mixer: [False, True]
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
import matplotlib.pyplot as plt

from algo_data_loader import AmazonDataLoader
from algo_hybrid_LRWSQAOA import run_algo_hybrid_2_5


def compute_open_route_cost(tour, matrix):
    """Calculates Open TSP cost (accumulated travel time along sequence without return to depot)."""
    return float(sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


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


def generate_overall_visualizations(data, hybrid_tour, hybrid_cost, param_str, output_dir):
    """
    Generates side-by-side comparison plots between Amazon Planned and Hybrid QAOA.
    Outputs to separate folders:
      1. `plots_with_depot/`: Route layout including Depot
      2. `plots_without_depot/`: Delivery stops layout only
    """
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

    filename_slug = f"{route_id}_{param_str}.png"

    # ------------------ 1. PLOT WITH DEPOT ------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Amazon Planned (With Depot)
    ax = axes[0]
    p_amazon = coords[amazon_tour]
    ax.plot(p_amazon[:, 0], p_amazon[:, 1], "b-o", linewidth=2, label=f"Amazon Planned ({amazon_cost:.2f})")
    ax.scatter(coords[depot_idx, 0], coords[depot_idx, 1], c="red", s=180, marker="D", label="Depot", zorder=5)
    for i in range(len(coords)):
        ax.annotate(f" {i}", (coords[i, 0], coords[i, 1]), fontsize=9, weight="bold")
    ax.set_title("Amazon Planned Route (With Depot)", fontsize=11, weight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Hybrid QAOA 2+5 (With Depot)
    ax = axes[1]
    p_hybrid = coords[hybrid_tour]
    ax.plot(p_hybrid[:, 0], p_hybrid[:, 1], "g-o", linewidth=2, label=f"Hybrid 2+5 ({hybrid_cost:.2f})")
    ax.scatter(coords[depot_idx, 0], coords[depot_idx, 1], c="red", s=180, marker="D", label="Depot", zorder=5)
    for i in range(len(coords)):
        ax.annotate(f" {i}", (coords[i, 0], coords[i, 1]), fontsize=9, weight="bold")
    ax.set_title("Hybrid Algo 2+5 Route (With Depot)", fontsize=11, weight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(f"Route {route_id} (With Depot) | Params: {param_str}", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(depot_dir, filename_slug), dpi=120)
    plt.close(fig)

    # ------------------ 2. PLOT WITHOUT DEPOT ------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    delivery_indices = [i for i in range(len(coords)) if i != depot_idx]

    # Amazon Planned (Without Depot)
    ax = axes[0]
    amazon_no_depot = [i for i in amazon_tour if i != depot_idx]
    p_amazon_nd = coords[amazon_no_depot]
    ax.plot(p_amazon_nd[:, 0], p_amazon_nd[:, 1], "b-o", linewidth=2, label=f"Amazon Planned ({amazon_cost:.2f})")
    for i in delivery_indices:
        ax.annotate(f" {i}", (coords[i, 0], coords[i, 1]), fontsize=9, weight="bold")
    ax.set_title("Amazon Planned Route (Delivery Stops Only)", fontsize=11, weight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Hybrid QAOA 2+5 (Without Depot)
    ax = axes[1]
    hybrid_no_depot = [i for i in hybrid_tour if i != depot_idx]
    p_hybrid_nd = coords[hybrid_no_depot]
    ax.plot(p_hybrid_nd[:, 0], p_hybrid_nd[:, 1], "g-o", linewidth=2, label=f"Hybrid 2+5 ({hybrid_cost:.2f})")
    for i in delivery_indices:
        ax.annotate(f" {i}", (coords[i, 0], coords[i, 1]), fontsize=9, weight="bold")
    ax.set_title("Hybrid Algo 2+5 Route (Delivery Stops Only)", fontsize=11, weight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(f"Route {route_id} (Without Depot) | Params: {param_str}", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(no_depot_dir, filename_slug), dpi=120)
    plt.close(fig)

    plt.close("all")
    gc.collect()


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
            cost_diff_pct = (cost_diff_abs / amazon_cost) * 100.0
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
                generate_overall_visualizations(data, hybrid_tour, hybrid_cost, param_str, output_dir)

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
    parser = argparse.ArgumentParser(description="Run Hybrid QAOA 2+5 Parameter Grid Search vs Amazon Baseline")
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training", help="Dataset directory")
    parser.add_argument("--num-routes", type=int, default=100, help="Number of routes to sample (default: 10)")
    parser.add_argument("--output-dir", type=str, default="./experiment_results", help="Output directory for CSVs & plots")
    parser.add_argument("--no-plots", action="store_true", help="Disable generating visual plots to speed up runs")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")

    args = parser.parse_args()

    run_experiment_grid(
        data_dir=args.data_dir,
        num_routes=args.num_routes,
        qubit_counts=[2, 3],
        exploration_percents=[0.0, 0.2],
        batch_counts=[1, 2, 3, 4],
        # xy_mixers=[False, True],
        xy_mixers=[False],
        output_dir=args.output_dir,
        seed=args.seed,
        generate_plots=not args.no_plots,
    )