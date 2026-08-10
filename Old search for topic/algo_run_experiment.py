"""
Main Experiment Harness comparing Algo 1 through Algo 6.
Fixed: Evaluates Amazon Planned Route directly through compute_tour_minutes()
for a fair apples-to-apples transit metric comparison.
"""

import os
import random
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from algo1_rolling_pca_lns import run_algo1
from algo2_rolling_lns import run_algo2
from algo3_spectral_lns import run_algo3
from algo4_relational_non_island_qa import run_algo4
from algo5_wslr_qaoa_batch import run_algo5
from algo6_global_lr_graph_qaoa import run_algo6
from algo_data_loader import AmazonDataLoader

# Configuration Flags
FULL_DATASET = False
OUTPUT_DIR = "./visualizations"
QUBIT_BUDGET = 5


def compute_tour_minutes(matrix, tour):
    """Calculates total duration in minutes for an Open TSP route using pure travel time matrix."""
    if not tour or len(tour) < 2:
        return 0.0
    total_sec = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    return total_sec / 60.0


def plot_and_save_routes(
    route_id, matrix, coords, depot_idx, planned_seq, planned_cost_min, algos_results
):
    """Generates visual comparison plots (both with and without depot legs)."""
    route_folder = os.path.join(OUTPUT_DIR, route_id)
    os.makedirs(route_folder, exist_ok=True)

    non_depot_mask = np.ones(len(coords), dtype=bool)
    non_depot_mask[depot_idx] = False
    delivery_coords = coords[non_depot_mask]

    # Pre-calculate Amazon Stops-Only Sequence & Cost
    planned_no_depot = [i for i in planned_seq if i != depot_idx and i < len(coords)]
    planned_stops_cost_min = compute_tour_minutes(matrix, planned_no_depot)

    for name, seq, cost_min in algos_results:
        clean_name = name.split()[0].replace(":", "").strip()

        # -----------------------------------------------------------------
        # PLOT 1: Full Route (Depot Included)
        # -----------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

        pts_p = coords[[i for i in planned_seq if i < len(coords)]]
        axes[0].plot(pts_p[:, 1], pts_p[:, 0], color="#1E88E5", linewidth=1.5, alpha=0.85)
        axes[0].scatter(delivery_coords[:, 1], delivery_coords[:, 0], c="black", s=20, zorder=3)
        axes[0].scatter(coords[depot_idx, 1], coords[depot_idx, 0], c="red", s=80, marker="^", zorder=4)
        axes[0].set_title(f"Amazon Planned Route\nTravel Cost: {planned_cost_min:.2f} min", fontweight="bold")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        pts_a = coords[seq]
        diff_pct = (((cost_min - planned_cost_min) / planned_cost_min) * 100.0) if planned_cost_min > 0 else 0.0
        axes[1].plot(pts_a[:, 1], pts_a[:, 0], color="#D9381E", linewidth=1.5, alpha=0.85)
        axes[1].scatter(delivery_coords[:, 1], delivery_coords[:, 0], c="black", s=20, zorder=3)
        axes[1].scatter(coords[depot_idx, 1], coords[depot_idx, 0], c="red", s=80, marker="^", zorder=4)
        axes[1].set_title(f"{name}\nTravel Cost: {cost_min:.2f} min ({diff_pct:+.2f}% vs Planned)", fontweight="bold")
        axes[1].set_xlabel("Longitude")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.suptitle(f"Route: {route_id} | {name} vs Amazon Planned (With Depot)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(route_folder, f"comparison_{clean_name}.png"), dpi=200)
        plt.close(fig)

        # -----------------------------------------------------------------
        # PLOT 2: Delivery Only (Depot Excluded)
        # -----------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

        pts_p_no_depot = coords[planned_no_depot]
        axes[0].plot(pts_p_no_depot[:, 1], pts_p_no_depot[:, 0], color="#1E88E5", linewidth=1.5, alpha=0.85)
        axes[0].scatter(delivery_coords[:, 1], delivery_coords[:, 0], c="black", s=25, zorder=3)
        axes[0].set_title(f"Amazon Planned (Stops Only)\nCost: {planned_stops_cost_min:.2f} min", fontweight="bold")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        algo_no_depot = [i for i in seq if i != depot_idx]
        algo_stops_cost_min = compute_tour_minutes(matrix, algo_no_depot)
        diff_stops_pct = (((algo_stops_cost_min - planned_stops_cost_min) / planned_stops_cost_min) * 100.0) if planned_stops_cost_min > 0 else 0.0

        pts_a_no_depot = coords[algo_no_depot]
        axes[1].plot(pts_a_no_depot[:, 1], pts_a_no_depot[:, 0], color="#2E7D32", linewidth=1.5, alpha=0.85)
        axes[1].scatter(delivery_coords[:, 1], delivery_coords[:, 0], c="black", s=25, zorder=3)
        axes[1].set_title(f"{name} (Stops Only)\nCost: {algo_stops_cost_min:.2f} min ({diff_stops_pct:+.2f}% vs Planned)", fontweight="bold")
        axes[1].set_xlabel("Longitude")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.suptitle(f"Route: {route_id} | {name} vs Amazon Planned (EXCLUDING DEPOT)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(route_folder, f"no_depot_comparison_{clean_name}.png"), dpi=200)
        plt.close(fig)


def run_experiment_suite():
    loader = AmazonDataLoader()
    available_routes = list(loader.travel_times.keys())
    total_count = len(available_routes)

    print("=" * 80)
    print(f"DATASET LOADED: Total available routes = {total_count}")
    print(f"FULL_DATASET mode = {FULL_DATASET} | Qubits = {QUBIT_BUDGET}")
    print("=" * 80)

    if not available_routes:
        print("No routes found in travel_times.json!")
        return

    routes_to_process = (
        available_routes if FULL_DATASET else [random.choice(available_routes)]
    )

    algos = [
        ("Algo1 (Rolling + PCA + LNS)", run_algo1),
        ("Algo2 (Rolling + LNS)", run_algo2),
        ("Algo3 (Spectral + LNS)", run_algo3),
        ("Algo4 (Relational Non-Island QA)", run_algo4),
        ("Algo5 (WS-LR QAOA Batch + LNS)", run_algo5),
        ("Algo6 (Global LR Graph QAOA)", run_algo6),
    ]

    summary_table = []

    for idx, r_id in enumerate(routes_to_process, 1):
        print(f"\n[{idx}/{len(routes_to_process)}] Processing Route: {r_id}")
        data = loader.extract_single_route(r_id)

        coords = data["coords"]
        depot_idx = data["depot_idx"]
        planned_seq = data["amazon_planned_sequence"]
        num_stops = data["n_nodes"]
        matrix = data["matrix"]

        # Compute Amazon planned transit cost using the same cost function as the algorithms
        amazon_travel_cost_min = compute_tour_minutes(matrix, planned_seq)
        raw_amazon_meta_cost = float(data.get("amazon_planned_cost", 0.0))

        print(f"  -> Amazon Planned Matrix Travel Time: {amazon_travel_cost_min:.2f} min (Raw Dataset Meta Cost: {raw_amazon_meta_cost:.2f} min)")

        algos_results = []

        for algo_name, algo_fn in algos:
            start_t = time.time()
            try:
                res = algo_fn(data, num_qubits=QUBIT_BUDGET)
                elapsed = time.time() - start_t

                if isinstance(res, dict):
                    seq = res.get("tour") or res.get("route")
                    raw_cost = res.get("cost")
                else:
                    seq = res
                    raw_cost = None

                cost_min = (
                    compute_tour_minutes(matrix, seq)
                    if seq is not None
                    else (raw_cost / 60.0)
                )

                diff_pct = (
                    ((cost_min - amazon_travel_cost_min) / amazon_travel_cost_min) * 100.0
                    if amazon_travel_cost_min > 0
                    else 0.0
                )

                summary_table.append(
                    [
                        r_id[:16],
                        num_stops,
                        algo_name,
                        f"{cost_min:.2f} m",
                        f"{amazon_travel_cost_min:.2f} m",
                        f"{diff_pct:+.2f}%",
                        f"{elapsed:.2f}s",
                    ]
                )
                algos_results.append((algo_name, seq, cost_min))
            except Exception as e:
                print(f"Error in {algo_name}: {e}")
                summary_table.append(
                    [
                        r_id[:16],
                        num_stops,
                        algo_name,
                        "FAILED",
                        f"{amazon_travel_cost_min:.2f} m",
                        "N/A",
                        f"{time.time() - start_t:.2f}s",
                    ]
                )

        plot_and_save_routes(
            r_id,
            matrix,
            coords,
            depot_idx,
            planned_seq,
            amazon_travel_cost_min,
            algos_results,
        )

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS vs AMAZON PLANNED ROUTES (MATRIX TRAVEL TIME IN MINUTES)")
    print("=" * 80)
    headers = [
        "Route ID",
        "Stops",
        "Algorithm",
        "Algo Cost",
        "Amazon Planned",
        "vs Planned (%)",
        "Runtime",
    ]
    print(tabulate(summary_table, headers=headers, tablefmt="grid"))
    print(f"\nSaved all path plots to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run_experiment_suite()