"""
compare_amazon_planned.py

Comparative analysis script between Amazon Planned Route baseline and
Hybrid Algorithm 2+5 (WS-LR QAOA + LNS) using the real Amazon Last Mile
Routing Research Challenge dataset parsed via AmazonDataLoader.

Evaluates Open TSP (Without Depot Return).
"""

# --- CRITICAL CPU & THERMAL LIMITS ---
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import gc
import math
import random
import time
import numpy as np

# Non-interactive backend to prevent GUI blocking
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algo_hybrid_LRWSQAOA import solve_wslr_qaoa_subtour, run_algo_hybrid_2_5
from algo_data_loader import AmazonDataLoader


def get_real_amazon_delivery_dataset(data_dir="./almrrc2021-data-training", route_id=None):
    """
    Loads real Amazon Last Mile Routing Challenge data using AmazonDataLoader.
    Extracts travel times matrix, stop coordinates, and actual planned sequence.
    """
    if not os.path.exists(data_dir) and os.path.exists("./data"):
        data_dir = "./data"

    loader = AmazonDataLoader(data_dir=data_dir)
    
    if route_id is None:
        if loader.travel_times:
            route_id = list(loader.travel_times.keys())[0]
        else:
            raise FileNotFoundError(
                f"No route data found in '{data_dir}'. Ensure build_inputs or data_dir contains travel_times.json."
            )

    extracted = loader.extract_single_route(route_id)

    matrix = extracted["matrix"]
    coords = extracted["coords"]
    amazon_planned_tour = extracted["amazon_planned_sequence"]
    n_nodes = extracted["n_nodes"]

    if coords is None or np.all(coords == 0):
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=2026)
        coords = mds.fit_transform(matrix)

    return {
        "n_nodes": n_nodes,
        "coords": np.array(coords),
        "matrix": np.array(matrix),
        "depot_idx": extracted.get("depot_idx", 0),
        "amazon_planned_tour": amazon_planned_tour,
        "route_id": extracted.get("route_id", route_id),
    }


def compute_open_route_cost(tour, matrix):
    """Calculates Open TSP cost (accumulated travel cost along sequence without return to depot)."""
    return float(sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def visualize_stepwise_execution(
    data,
    qubit_count=4,
    exploration_percent=0.0,
    batch_count=1,
    xy_mixer=False,
    output_dir="qaoa_visualizations",
):
    """
    Executes Hybrid 2+5 step-by-step on real Amazon dataset (excluding depot from visuals).
    Enforces strict zero exploration when exploration_percent == 0.0.
    """
    os.makedirs(output_dir, exist_ok=True)

    coords = data["coords"]
    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data["depot_idx"]

    # Filter out depot node completely for visualization zoom
    delivery_indices = [i for i in range(n) if i != depot_idx]

    tour = [depot_idx]
    curr = depot_idx
    unvisited = set(range(n)) - {depot_idx}

    step_counter = 1
    t_start = time.time()

    print(
        f"=== Starting Step-by-Step Visualization Execution "
        f"(Qubits={qubit_count}, Exp={exploration_percent*100:.0f}%, Batch={batch_count}, "
        f"Route: {data['route_id']}) ==="
    )

    while unvisited:
        step_t0 = time.time()
        k_batch = min(qubit_count, len(unvisited))

        # Strict exploration budget allocation
        if exploration_percent <= 0.0:
            n_explore = 0
            n_nearest = k_batch
        else:
            n_explore = int(math.floor(k_batch * exploration_percent))
            if k_batch > 1 and n_explore >= k_batch:
                n_explore = k_batch - 1
            n_nearest = k_batch - n_explore

        # Deterministic sorting by travel cost, broken by node index
        sorted_unvisited = sorted(list(unvisited), key=lambda x: (matrix[curr, x], x))
        nearest_candidates = sorted_unvisited[:n_nearest]

        remaining_unvisited = sorted_unvisited[n_nearest:]
        if n_explore > 0 and remaining_unvisited:
            exploration_candidates = random.sample(
                remaining_unvisited, min(n_explore, len(remaining_unvisited))
            )
        else:
            exploration_candidates = []

        candidate_nodes = nearest_candidates + exploration_candidates

        full_qaoa_subtour = solve_wslr_qaoa_subtour(
            curr, candidate_nodes, matrix, xy_mixer=xy_mixer
        )

        commit_depth = min(batch_count, len(full_qaoa_subtour))
        nodes_to_commit = full_qaoa_subtour[:commit_depth]
        uncommitted_nodes = full_qaoa_subtour[commit_depth:]

        # Safety Fallback
        if not nodes_to_commit:
            fallback_node = sorted_unvisited[0]
            nodes_to_commit = [fallback_node]
            uncommitted_nodes = []
            full_qaoa_subtour = [fallback_node]

        # Plot Step Frame (EXCLUDING DEPOT TO PREVENT MAP DISTORTION)
        fig, ax = plt.subplots(figsize=(10, 8))

        # Only plot unvisited delivery stops
        unvisited_stops = [idx for idx in unvisited if idx != depot_idx]
        if unvisited_stops:
            unvisited_coords = coords[unvisited_stops]
            ax.scatter(
                unvisited_coords[:, 0],
                unvisited_coords[:, 1],
                c="lightgray",
                edgecolors="black",
                s=90,
                label="Unvisited Delivery Stops",
                zorder=2,
            )

        if nearest_candidates:
            nc = coords[nearest_candidates]
            ax.scatter(nc[:, 0], nc[:, 1], c="blue", s=140, marker="o", label="QAOA Nearest", zorder=3)
        if exploration_candidates:
            ec = coords[exploration_candidates]
            ax.scatter(ec[:, 0], ec[:, 1], c="orange", s=140, marker="^", label="QAOA Explore", zorder=3)

        # Plot committed route minus depot connections
        tour_no_depot = [node for node in tour if node != depot_idx]
        if len(tour_no_depot) > 1:
            tour_coords = coords[tour_no_depot]
            ax.plot(tour_coords[:, 0], tour_coords[:, 1], "k-o", linewidth=2, label="Committed Route So Far", zorder=4)

        full_subtour_path = [curr] + full_qaoa_subtour
        full_subtour_path_no_depot = [node for node in full_subtour_path if node != depot_idx]
        if len(full_subtour_path_no_depot) > 1:
            full_subtour_coords = coords[full_subtour_path_no_depot]
            ax.plot(
                full_subtour_coords[:, 0],
                full_subtour_coords[:, 1],
                color="purple",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Full QAOA Subtour ({len(full_qaoa_subtour)} Nodes)",
                zorder=5,
            )

        if uncommitted_nodes:
            uncommitted_path = [nodes_to_commit[-1]] + uncommitted_nodes
            uncommitted_coords = coords[uncommitted_path]
            ax.plot(
                uncommitted_coords[:, 0],
                uncommitted_coords[:, 1],
                color="red",
                linestyle=":",
                linewidth=2.5,
                label=f"Discarded QAOA Tail ({len(uncommitted_nodes)} Nodes)",
                zorder=6,
            )

        committed_path = [curr] + nodes_to_commit
        committed_path_no_depot = [node for node in committed_path if node != depot_idx]
        if len(committed_path_no_depot) > 1:
            committed_coords = coords[committed_path_no_depot]
            ax.plot(committed_coords[:, 0], committed_coords[:, 1], "g-", linewidth=3.5, label=f"Committed Batch (size={len(nodes_to_commit)})", zorder=7)

        for idx in delivery_indices:
            ax.annotate(f"  {idx}", (coords[idx, 0], coords[idx, 1]), fontsize=11, weight="bold")

        # Auto-scale strictly around delivery nodes
        deliv_coords = coords[delivery_indices]
        pad_x = (deliv_coords[:, 0].max() - deliv_coords[:, 0].min()) * 0.05
        pad_y = (deliv_coords[:, 1].max() - deliv_coords[:, 1].min()) * 0.05
        ax.set_xlim(deliv_coords[:, 0].min() - pad_x, deliv_coords[:, 0].max() + pad_x)
        ax.set_ylim(deliv_coords[:, 1].min() - pad_y, deliv_coords[:, 1].max() + pad_y)

        ax.set_title(
            f"Step {step_counter}: Delivery Sequence (Excluding Depot) - {time.time()-step_t0:.2f}s",
            fontsize=12,
            weight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        frame_filename = os.path.join(output_dir, f"qaoa_step_{step_counter}.png")
        plt.savefig(frame_filename, dpi=120)
        plt.close(fig)
        plt.close("all")
        gc.collect()

        print(f"  [Step {step_counter}] Saved frame in {time.time()-step_t0:.2f}s")

        tour.extend(nodes_to_commit)
        for node in nodes_to_commit:
            unvisited.remove(node)
        curr = nodes_to_commit[-1]
        step_counter += 1

    print(f"--> Step visualizations saved to '{output_dir}/' in {time.time()-t_start:.2f}s\n")


def run_comparative_benchmark(
    qubit_count: int = 4,
    exploration_percent: float = 0.0,
    batch_count: int = 1,
    xy_mixer: bool = False,
    visualise_step_by_step: bool = True,
):
    # 1. Load Real Amazon Dataset
    data = get_real_amazon_delivery_dataset()
    coords = data["coords"]
    matrix = data["matrix"]
    depot_idx = data["depot_idx"]
    amazon_tour = data["amazon_planned_tour"]

    # 2. Step-by-step visual frame generation toggle
    if visualise_step_by_step:
        visualize_stepwise_execution(
            data,
            qubit_count=qubit_count,
            exploration_percent=exploration_percent,
            batch_count=batch_count,
            xy_mixer=xy_mixer,
        )

    # 3. Compute Baseline Open TSP Cost
    amazon_open_cost = compute_open_route_cost(amazon_tour, matrix)

    # 4. Execute Full Hybrid Algo 2+5
    hybrid_result = run_algo_hybrid_2_5(
        data,
        qubit_count=qubit_count,
        exploration_percent=exploration_percent,
        batch_count=batch_count,
        xy_mixer=xy_mixer,
        seed=2026,
    )
    hybrid_tour = hybrid_result["tour"]
    hybrid_open_cost = compute_open_route_cost(hybrid_tour, matrix)

    # 5. Summary Comparison
    print("=" * 70)
    print(f"   BENCHMARK COMPARISON: REAL AMAZON ROUTE ({data['route_id']}) vs. HYBRID QAOA   ")
    print("=" * 70)
    print(f"Amazon Planned Sequence : {amazon_tour}")
    print(f"Hybrid QAOA 2+5 Sequence: {hybrid_tour}\n")

    open_diff = ((hybrid_open_cost - amazon_open_cost) / amazon_open_cost) * 100

    print(
        f"OPEN TSP ROUTE COST (Without Depot Return):"
        f"\n   - Amazon Planned Cost : {amazon_open_cost:.2f}"
        f"\n   - Hybrid 2+5 Cost     : {hybrid_open_cost:.2f}"
        f"\n   - Cost Improvement    : {-open_diff:+.2f}%"
    )
    print("=" * 70)

    # 6. Final Comparative Plot (Single Row, 2 Subplots)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    delivery_indices = [i for i in range(len(coords)) if i != depot_idx]

    # Subplot 1: Amazon Planned
    ax = axes[0]
    amazon_no_depot = [i for i in amazon_tour if i != depot_idx]
    p_amazon = coords[amazon_no_depot]
    ax.plot(p_amazon[:, 0], p_amazon[:, 1], "b-o", linewidth=2, label=f"Amazon Planned ({amazon_open_cost:.2f})")
    for i in delivery_indices:
        ax.annotate(f" {i}", (coords[i, 0], coords[i, 1]), fontsize=10, weight="bold")
    ax.set_title("Amazon Planned (Delivery Path)", fontsize=12, weight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Hybrid Algo 2+5
    ax = axes[1]
    hybrid_no_depot = [i for i in hybrid_tour if i != depot_idx]
    p_hybrid = coords[hybrid_no_depot]
    ax.plot(p_hybrid[:, 0], p_hybrid[:, 1], "g-o", linewidth=2, label=f"Hybrid 2+5 ({hybrid_open_cost:.2f})")
    for i in delivery_indices:
        ax.annotate(f" {i}", (coords[i, 0], coords[i, 1]), fontsize=10, weight="bold")
    ax.set_title("Hybrid Algo 2+5 (Delivery Path)", fontsize=12, weight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(
        f"Real Amazon Dataset Benchmark ({data['route_id']}): Planned vs. Hybrid WS-LR QAOA + LNS",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig("amazon_vs_hybrid2_5_comparison.png", dpi=150)
    plt.close("all")
    gc.collect()
    print("--> Final comparison plot saved as 'amazon_vs_hybrid2_5_comparison.png'")


if __name__ == "__main__":
    # Parameters now pass through consistently across benchmark & visualizer
    run_comparative_benchmark(
        qubit_count=4,
        exploration_percent=0.0,
        batch_count=1,
        xy_mixer=False,
        visualise_step_by_step=True,
    )