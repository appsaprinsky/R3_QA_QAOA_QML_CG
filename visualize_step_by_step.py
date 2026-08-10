"""
visualize_step_by_step.py

Fast step-by-step execution visualizer for Hybrid Algorithm 2+5 (WS-LR QAOA + LNS).
Saves frames into 'qaoa_visualizations/' without GUI blocking.
"""

import os
import random
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent GUI thread lock
import matplotlib.pyplot as plt

from algo_hybrid_LRWSQAOA import solve_wslr_qaoa_subtour


def generate_random_tsp_data(n_nodes=12, seed=101):
    """Generates random 2D spatial Euclidean coordinates and distance matrix."""
    np.random.seed(seed)
    coords = np.random.uniform(10, 90, size=(n_nodes, 2))
    coords[0] = [50.0, 50.0]  # Depot

    matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    return {"n_nodes": n_nodes, "coords": coords, "matrix": matrix, "depot_idx": 0}


def visualize_stepwise_execution(
    data,
    qubit_count=4,  # Kept at 4 for fast simulation execution
    exploration_percent=0.25,
    batch_count=2,
    xy_mixer=False,
    seed=101,
    output_dir="qaoa_visualizations",
):
    """Executes Hybrid 2+5 step-by-step and saves frame visuals."""
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    coords = data["coords"]
    matrix = data["matrix"]
    n = data["n_nodes"]
    depot_idx = data["depot_idx"]

    tour = [depot_idx]
    curr = depot_idx
    unvisited = set(range(n)) - {depot_idx}

    step_counter = 1
    t_start = time.time()

    while unvisited:
        step_t0 = time.time()
        k_batch = min(qubit_count, len(unvisited))
        n_explore = int(round(k_batch * exploration_percent))
        n_explore = min(n_explore, k_batch - 1) if k_batch > 1 else 0
        n_nearest = k_batch - n_explore

        sorted_unvisited = sorted(list(unvisited), key=lambda x: matrix[curr, x])
        nearest_candidates = sorted_unvisited[:n_nearest]

        remaining_unvisited = sorted_unvisited[n_nearest:]
        if n_explore > 0 and remaining_unvisited:
            exploration_candidates = random.sample(
                remaining_unvisited, min(n_explore, len(remaining_unvisited))
            )
        else:
            exploration_candidates = []

        candidate_nodes = nearest_candidates + exploration_candidates

        # Solve QAOA subtour
        full_qaoa_subtour = solve_wslr_qaoa_subtour(
            curr, candidate_nodes, matrix, xy_mixer=xy_mixer
        )

        commit_depth = min(batch_count, len(full_qaoa_subtour))
        nodes_to_commit = full_qaoa_subtour[:commit_depth]
        uncommitted_nodes = full_qaoa_subtour[commit_depth:]

        # --- PLOT STEP FRAME ---
        fig, ax = plt.subplots(figsize=(9, 7))

        # Unvisited nodes
        unvisited_coords = coords[list(unvisited)]
        ax.scatter(
            unvisited_coords[:, 0],
            unvisited_coords[:, 1],
            c="gray",
            s=80,
            label="Unvisited Nodes",
            zorder=2,
        )

        # Candidates
        if nearest_candidates:
            nc = coords[nearest_candidates]
            ax.scatter(
                nc[:, 0], nc[:, 1], c="blue", s=140, marker="o", label="QAOA Nearest", zorder=3
            )
        if exploration_candidates:
            ec = coords[exploration_candidates]
            ax.scatter(
                ec[:, 0], ec[:, 1], c="orange", s=140, marker="^", label="QAOA Explore", zorder=3
            )

        # Committed Tour Path
        if len(tour) > 1:
            tour_coords = coords[tour]
            ax.plot(
                tour_coords[:, 0],
                tour_coords[:, 1],
                "k-o",
                linewidth=2,
                label="Committed Tour",
                zorder=4,
            )

        # Full QAOA Subtour Solution
        full_subtour_path = [curr] + full_qaoa_subtour
        full_subtour_coords = coords[full_subtour_path]
        ax.plot(
            full_subtour_coords[:, 0],
            full_subtour_coords[:, 1],
            color="purple",
            linestyle="--",
            linewidth=2,
            label="Full QAOA Subtour",
            zorder=5,
        )

        # Committed Batch Step
        committed_coords = coords[[curr] + nodes_to_commit]
        ax.plot(
            committed_coords[:, 0],
            committed_coords[:, 1],
            "g-",
            linewidth=3.5,
            label=f"Commit Batch ({commit_depth})",
            zorder=6,
        )

        # Depot
        ax.scatter(
            coords[depot_idx, 0],
            coords[depot_idx, 1],
            c="green",
            s=220,
            marker="s",
            label="Depot (Node 0)",
            zorder=7,
        )

        for idx in range(n):
            ax.annotate(
                f"  {idx}", (coords[idx, 0], coords[idx, 1]), fontsize=11, weight="bold"
            )

        ax.set_title(
            f"Step {step_counter} (Qubits={k_batch}, Commit={commit_depth}) - Done in {time.time()-step_t0:.2f}s",
            fontsize=12,
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        frame_file = os.path.join(output_dir, f"qaoa_step_{step_counter}.png")
        plt.savefig(frame_file, dpi=150)
        plt.close(fig)

        print(f"Step {step_counter} saved ({time.time()-step_t0:.2f}s)")

        # Update tour state
        tour.extend(nodes_to_commit)
        for node in nodes_to_commit:
            unvisited.remove(node)
        curr = nodes_to_commit[-1]
        step_counter += 1

    # Final Route Frame
    fig, ax = plt.subplots(figsize=(9, 7))
    tour_coords = coords[tour]
    ax.plot(tour_coords[:, 0], tour_coords[:, 1], "g-o", linewidth=2.5, label="Final Route")
    ax.scatter(coords[depot_idx, 0], coords[depot_idx, 1], c="red", s=220, marker="s", label="Depot")
    for idx in range(n):
        ax.annotate(f"  {idx}", (coords[idx, 0], coords[idx, 1]), fontsize=11, weight="bold")
    ax.set_title("Final Constructed Tour", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qaoa_final_tour.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll frames generated in {time.time()-t_start:.2f}s under '{output_dir}/'")


if __name__ == "__main__":
    data = generate_random_tsp_data(n_nodes=12, seed=101)
    visualize_stepwise_execution(
        data, qubit_count=4, exploration_percent=0.25, batch_count=2, xy_mixer=False
    )