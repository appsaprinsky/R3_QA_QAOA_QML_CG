"""
qubit_sensitivity_benchmark_algo2.py

Qubit Budget Sensitivity Benchmark for Algorithm 2 (WS-LR QAOA Batch Optimization + LNS).
Directly imports `run_algo2` from `algo2_wslr_qaoa_batch.py` without modifying or
re-implementing any algorithm logic.

Generates:
1. Benchmark performance summary table across qubit counts (k = 2 to 10).
2. Qubit count vs Travel Time and Execution Time sensitivity plot.
3. Final delivery-only route map visualizations (EXCLUDING DEPOT) for each qubit count configuration vs Amazon Planned.
"""

import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# DIRECT IMPORT FROM ALGORITHM FILE
# from algo2_rolling_lns import run_algo2
# from algo6_global_lr_graph_qaoa import run_algo6 as run_algo2
from algo5_wslr_qaoa_batch import run_algo5 as run_algo2
from algo_data_loader import AmazonDataLoader


def compute_stops_only_cost_sec(matrix, tour_no_depot):
    """Calculates cumulative travel time in seconds for a sequence excluding depot."""
    if len(tour_no_depot) < 2:
        return 0.0
    return sum(
        matrix[tour_no_depot[i], tour_no_depot[i + 1]]
        for i in range(len(tour_no_depot) - 1)
    )


def benchmark_qubits_algo2(
    data, qubit_range=range(2, 11), output_dir="qubit_sensitivity_algo2_results"
):
    """
    Sweeps qubit capacities k across `qubit_range` using directly imported `run_algo2`.
    """
    os.makedirs(output_dir, exist_ok=True)

    matrix = data["matrix"]
    coords = np.array(data["coords"])
    n = data["n_nodes"]
    depot_idx = data["depot_idx"]
    route_id = data.get("route_id", "benchmark_route")

    # Amazon Planned Route (Full and No-Depot)
    planned_seq = data["amazon_planned_sequence"]
    planned_full_cost_sec = sum(
        matrix[planned_seq[i], planned_seq[i + 1]]
        for i in range(len(planned_seq) - 1)
    )
    planned_full_cost_min = planned_full_cost_sec / 60.0

    planned_no_depot = [
        node for node in planned_seq if node != depot_idx and node < len(coords)
    ]
    planned_stops_cost_sec = compute_stops_only_cost_sec(
        matrix, planned_no_depot
    )
    planned_stops_cost_min = planned_stops_cost_sec / 60.0

    results = []

    print("=" * 90)
    print(
        f"BENCHMARKING DIRECT IMPORT `run_algo2` ACROSS QUBITS k = {list(qubit_range)}"
    )
    print(f"Route ID: {route_id} | Total Stops: {n}")
    print(
        f"Amazon Planned Full Cost: {planned_full_cost_min:.2f} min | Stops Only Cost: {planned_stops_cost_min:.2f} min"
    )
    print("=" * 90)

    for q in qubit_range:
        start_t = time.time()

        # DIRECT CALL TO IMPORTED ALGORITHM FUNCTION
        res = run_algo2(data, num_qubits=q)

        elapsed = time.time() - start_t

        algo_full_tour = res["tour"]
        algo_full_cost_sec = res["cost"]
        algo_full_cost_min = algo_full_cost_sec / 60.0
        diff_full_pct = (
            (algo_full_cost_min - planned_full_cost_min) / planned_full_cost_min
        ) * 100.0

        # Extract delivery-only tour (Excluding Depot)
        algo_no_depot = [node for node in algo_full_tour if node != depot_idx]
        algo_stops_cost_sec = compute_stops_only_cost_sec(
            matrix, algo_no_depot
        )
        algo_stops_cost_min = algo_stops_cost_sec / 60.0
        diff_stops_pct = (
            (algo_stops_cost_min - planned_stops_cost_min)
            / planned_stops_cost_min
        ) * 100.0

        results.append(
            {
                "qubits": q,
                "full_tour": algo_full_tour,
                "no_depot_tour": algo_no_depot,
                "full_cost_min": algo_full_cost_min,
                "stops_cost_min": algo_stops_cost_min,
                "diff_full_pct": diff_full_pct,
                "diff_stops_pct": diff_stops_pct,
                "runtime_sec": elapsed,
            }
        )

        print(
            f"Qubits (k) = {q:2d} | Full Cost: {algo_full_cost_min:.2f} m ({diff_full_pct:+.2f}%) | "
            f"Stops Only: {algo_stops_cost_min:.2f} m ({diff_stops_pct:+.2f}%) | Time: {elapsed:.3f}s"
        )

    # Output Summary Table
    table_data = []
    for r in results:
        table_data.append(
            [
                r["qubits"],
                f"{r['full_cost_min']:.2f} m",
                f"{r['stops_cost_min']:.2f} m",
                f"{planned_stops_cost_min:.2f} m",
                f"{r['diff_stops_pct']:+.2f}%",
                f"{r['runtime_sec']:.3f} s",
            ]
        )

    headers = [
        "Qubits (k)",
        "Full Tour Cost",
        "Algo Stops Cost",
        "Amazon Stops Cost",
        "vs Amazon (%)",
        "Runtime",
    ]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    # 1. Plot Sensitivity Curve
    qubit_list = [r["qubits"] for r in results]
    stops_costs = [r["stops_cost_min"] for r in results]
    runtimes = [r["runtime_sec"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        qubit_list,
        stops_costs,
        color="#00695C",
        marker="o",
        linewidth=2.2,
        label="Algo 2 Stops Cost (min)",
    )
    l2 = ax1.axhline(
        planned_stops_cost_min,
        color="#D32F2F",
        linestyle="--",
        linewidth=1.8,
        label="Amazon Planned Stops Cost",
    )
    l3 = ax2.plot(
        qubit_list,
        runtimes,
        color="#7B1FA2",
        marker="s",
        linestyle=":",
        linewidth=1.8,
        label="Runtime (s)",
    )

    ax1.set_xlabel("Qubit Capacity (k Candidate Batch Size)", fontweight="bold")
    ax1.set_ylabel(
        "Travel Time - Stops Only (minutes)", fontweight="bold", color="#00695C"
    )
    ax2.set_ylabel("Execution Time (seconds)", fontweight="bold", color="#7B1FA2")
    ax1.set_xticks(qubit_list)
    ax1.grid(True, linestyle=":", alpha=0.6)

    lines = l1 + [l2] + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.title(
        f"Algorithm 2 Direct-Import Qubit Sensitivity Analysis (Route: {route_id[:16]})",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qubit_sensitivity_curve_algo2.png"), dpi=200)
    plt.close(fig)

    # 2. Plot Delivery-Only Route Comparisons (WITHOUT DEPOT)
    non_depot_mask = np.ones(n, dtype=bool)
    non_depot_mask[depot_idx] = False
    delivery_coords = coords[non_depot_mask]
    pts_p_no_depot = coords[planned_no_depot]

    maps_folder = os.path.join(output_dir, "no_depot_route_maps_algo2")
    os.makedirs(maps_folder, exist_ok=True)

    for r in results:
        q = r["qubits"]
        algo_no_depot = r["no_depot_tour"]
        stops_cost_min = r["stops_cost_min"]
        diff_stops_pct = r["diff_stops_pct"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

        # Left: Amazon Planned (Stops Only)
        axes[0].plot(
            pts_p_no_depot[:, 1],
            pts_p_no_depot[:, 0],
            color="#1E88E5",
            linewidth=1.5,
            alpha=0.85,
        )
        axes[0].scatter(
            delivery_coords[:, 1],
            delivery_coords[:, 0],
            c="black",
            s=25,
            zorder=3,
        )
        axes[0].set_title(
            f"Amazon Planned (Stops Only)\nCost: {planned_stops_cost_min:.2f} min",
            fontweight="bold",
        )
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # Right: Direct Algo 2 (Stops Only) for Qubit q
        pts_a_no_depot = coords[algo_no_depot]
        axes[1].plot(
            pts_a_no_depot[:, 1],
            pts_a_no_depot[:, 0],
            color="#00796B",
            linewidth=1.5,
            alpha=0.85,
        )
        axes[1].scatter(
            delivery_coords[:, 1],
            delivery_coords[:, 0],
            c="black",
            s=25,
            zorder=3,
        )
        axes[1].set_title(
            f"Algo 2 (Qubits k={q}, Stops Only)\nCost: {stops_cost_min:.2f} min ({diff_stops_pct:+.2f}% vs Planned)",
            fontweight="bold",
        )
        axes[1].set_xlabel("Longitude")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        plt.suptitle(
            f"Route: {route_id[:16]} | Algo 2 Qubits k={q} vs Amazon Planned (EXCLUDING DEPOT)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(maps_folder, f"no_depot_k{q:02d}.png"), dpi=200
        )
        plt.close(fig)

    print(
        f"\nSaved Sensitivity Curve & {len(results)} No-Depot Route Maps to: '{os.path.abspath(output_dir)}'"
    )


if __name__ == "__main__":
    loader = AmazonDataLoader()
    available_routes = list(loader.travel_times.keys())

    if not available_routes:
        print("No routes found in travel_times.json!")
    else:
        selected_route_id = random.choice(available_routes)
        data = loader.extract_single_route(selected_route_id)
        data["route_id"] = selected_route_id

        # Evaluates k in [2..10].
        benchmark_qubits_algo2(
            data, qubit_range=range(2, 11), output_dir="qubit_sensitivity_algo2_results"
        )