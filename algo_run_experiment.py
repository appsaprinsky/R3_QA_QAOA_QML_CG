"""
Main Experiment Harness.
Imports separate algorithm modules (algo1, algo2, algo3, algo4).
Runs full Amazon Planned route benchmarks and saves comparative path plots.
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
from algo_data_loader import AmazonDataLoader

# Configuration Flags
FULL_DATASET = False  # True to run all routes, False for 1 random route
OUTPUT_DIR = "./visualizations"
QUBIT_BUDGET = 5


def plot_and_save_routes(
    route_id, coords, depot_idx, planned_seq, planned_cost, algos_results
):
    """Generates and saves visual plots comparing each algorithm against Amazon Planned."""
    route_folder = os.path.join(OUTPUT_DIR, route_id)
    os.makedirs(route_folder, exist_ok=True)

    non_depot_mask = np.ones(len(coords), dtype=bool)
    non_depot_mask[depot_idx] = False
    delivery_coords = coords[non_depot_mask]

    for name, seq, cost in algos_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

        # Plot Amazon Planned
        pts_p = coords[[i for i in planned_seq if i < len(coords)]]
        axes[0].plot(
            pts_p[:, 1], pts_p[:, 0], color="#1E88E5", linewidth=1.5, alpha=0.85
        )
        axes[0].scatter(
            delivery_coords[:, 1],
            delivery_coords[:, 0],
            c="black",
            s=20,
            zorder=3,
        )
        axes[0].scatter(
            coords[depot_idx, 1],
            coords[depot_idx, 0],
            c="red",
            s=80,
            marker="^",
            zorder=4,
        )
        axes[0].set_title(
            f"Amazon Planned Route\nCost: {planned_cost:.2f}", fontweight="bold"
        )
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # Plot Algorithm Path
        pts_a = coords[seq]
        diff_pct = (
            ((cost - planned_cost) / planned_cost) * 100.0
            if planned_cost > 0
            else 0.0
        )
        axes[1].plot(
            pts_a[:, 1], pts_a[:, 0], color="#D9381E", linewidth=1.5, alpha=0.85
        )
        axes[1].scatter(
            delivery_coords[:, 1],
            delivery_coords[:, 0],
            c="black",
            s=20,
            zorder=3,
        )
        axes[1].scatter(
            coords[depot_idx, 1],
            coords[depot_idx, 0],
            c="red",
            s=80,
            marker="^",
            zorder=4,
        )
        axes[1].set_title(
            f"{name}\nCost: {cost:.2f} ({diff_pct:+.2f}% vs Planned)",
            fontweight="bold",
        )
        axes[1].set_xlabel("Longitude")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        clean_name = name.split()[0].replace(":", "").strip()
        plt.suptitle(
            f"Route: {route_id} | {name} vs Amazon Planned",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(route_folder, f"comparison_{clean_name}.png"), dpi=200
        )
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
    ]

    summary_table = []

    for idx, r_id in enumerate(routes_to_process, 1):
        print(f"\n[{idx}/{len(routes_to_process)}] Processing Route: {r_id}")
        data = loader.extract_single_route(r_id)

        amazon_cost = data["amazon_planned_cost"]
        coords = data["coords"]
        depot_idx = data["depot_idx"]
        planned_seq = data["amazon_planned_sequence"]
        num_stops = data["n_nodes"]

        algos_results = []

        for algo_name, algo_fn in algos:
            start_t = time.time()
            try:
                # Pass data dict to modular algorithm functions
                res = algo_fn(data, num_qubits=QUBIT_BUDGET)
                elapsed = time.time() - start_t

                cost = res["cost"] if isinstance(res, dict) else res
                seq = res["route"] if isinstance(res, dict) else res

                diff_pct = (
                    ((cost - amazon_cost) / amazon_cost) * 100.0
                    if amazon_cost > 0
                    else 0.0
                )
                summary_table.append(
                    [
                        r_id[:16],
                        num_stops,
                        algo_name,
                        f"{cost:.2f}",
                        f"{amazon_cost:.2f}",
                        f"{diff_pct:+.2f}%",
                        f"{elapsed:.2f}s",
                    ]
                )
                algos_results.append((algo_name, seq, cost))
            except Exception as e:
                summary_table.append(
                    [
                        r_id[:16],
                        num_stops,
                        algo_name,
                        "FAILED",
                        f"{amazon_cost:.2f}",
                        "N/A",
                        f"{time.time() - start_t:.2f}s",
                    ]
                )

        plot_and_save_routes(
            r_id,
            coords,
            depot_idx,
            planned_seq,
            amazon_cost,
            algos_results,
        )

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS vs AMAZON PLANNED ROUTES")
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