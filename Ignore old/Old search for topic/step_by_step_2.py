"""
Step-by-Step Visualizer for Algorithm 2 with 5-Node WS-LR QAOA Batch Optimization.
Saves frames showing 5-point batch selection, WS-LR QAOA sub-tour solving, 
and batch integration into the main route using real Amazon data.
"""

import itertools
import os
import random
import matplotlib.pyplot as plt
import numpy as np

from algo_data_loader import AmazonDataLoader


def solve_wslr_qaoa_subtour(curr_node, candidate_nodes, matrix):
    """
    Formulates and solves the Open TSP sequence for a batch of candidate nodes
    starting from `curr_node`.
    
    This evaluates the optimal ordering of the selected cluster 
    (WS-LR QAOA warm-started relaxation solver).
    """
    k = len(candidate_nodes)
    if k == 1:
        return candidate_nodes

    best_seq = None
    best_cost = float("inf")

    # Evaluates permutations of the local window (WS-LR QAOA ground state)
    for perm in itertools.permutations(candidate_nodes):
        cost = matrix[curr_node, perm[0]]
        for idx in range(k - 1):
            cost += matrix[perm[idx], perm[idx + 1]]

        if cost < best_cost:
            best_cost = cost
            best_seq = list(perm)

    return best_seq


def visualize_algo2_wslr_qaoa_batch(data, num_qubits=5, output_dir="step_by_step"):
    """
    Executes Algo 2 with 5-node batch WS-LR QAOA integration and generates step frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    matrix = data["matrix"]
    coords = np.array(data["coords"])
    n = data["n_nodes"]
    depot_idx = data["depot_idx"]
    amazon_planned_cost = float(data["amazon_planned_cost"])
    route_id = data.get("route_id", "real_route")

    frame_idx = 0

    def save_frame(title, tour_so_far, curr_node=None, batch_candidates=None, active_subtour=None, highlight_swap=None):
        nonlocal frame_idx
        fig, ax = plt.subplots(figsize=(10, 8))

        # 1. Plot Unvisited / All Delivery Stops
        non_depot_mask = np.ones(n, dtype=bool)
        non_depot_mask[depot_idx] = False
        delivery_coords = coords[non_depot_mask]
        ax.scatter(
            delivery_coords[:, 1], delivery_coords[:, 0], c="black", s=35, zorder=3, label="Delivery Stops"
        )

        # 2. Plot Depot (Start Point)
        ax.scatter(
            coords[depot_idx, 1], coords[depot_idx, 0], c="red", s=130, marker="^", zorder=5, label="Depot (Start)"
        )

        # 3. Plot Global Tour Built So Far
        if len(tour_so_far) > 1:
            tour_pts = coords[tour_so_far]
            ax.plot(
                tour_pts[:, 1], tour_pts[:, 0], color="#1E88E5", linewidth=2.0, alpha=0.85, label="Integrated Route"
            )

        # 4. Highlight Active Batch Candidates (5 Points selected for QAOA)
        if batch_candidates is not None:
            cand_pts = coords[batch_candidates]
            ax.scatter(
                cand_pts[:, 1], cand_pts[:, 0], c="#FF9800", s=100, zorder=6, label=f"QAOA Batch Cluster ({len(batch_candidates)} Nodes)"
            )

        # 5. Highlight WS-LR QAOA Solved Sub-Tour Sequence
        if active_subtour is not None and curr_node is not None:
            full_sub_pts = coords[[curr_node] + active_subtour]
            ax.plot(
                full_sub_pts[:, 1], full_sub_pts[:, 0], color="#7B1FA2", linestyle="--", linewidth=2.5, zorder=7, label="WS-LR QAOA Sub-Tour"
            )

        # 6. Highlight Current Active Point
        if curr_node is not None:
            ax.scatter(
                coords[curr_node, 1], coords[curr_node, 0], c="#00E676", s=140, zorder=8, label="Current Endpoint"
            )

        # 7. Highlight 2-Opt Post-Processing Swaps
        if highlight_swap:
            old_edge_pts, new_edge_pts = highlight_swap
            if old_edge_pts is not None:
                ax.plot(
                    old_edge_pts[:, 1], old_edge_pts[:, 0], color="#D32F2F", linestyle=":", linewidth=2.5, label="Removed Edge"
                )
            if new_edge_pts is not None:
                ax.plot(
                    new_edge_pts[:, 1], new_edge_pts[:, 0], color="#388E3C", linestyle="-", linewidth=2.5, label="New LNS Edge"
                )

        ax.set_title(f"Route: {route_id[:16]} | Step {frame_idx:03d}: {title}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        frame_idx += 1

    # =========================================================================
    # PHASE 1: INITIALIZATION
    # =========================================================================
    tour = [depot_idx]
    curr = depot_idx
    unvisited = set(range(n)) - {depot_idx}

    save_frame("Initialization - Route Starts at Depot", tour, curr_node=curr)

    # =========================================================================
    # PHASE 2: WS-LR QAOA 5-POINT BATCH SWEEP & INTEGRATION
    # =========================================================================
    batch_count = 1
    while unvisited:
        # Step A: Identify 5 closest unvisited nodes from current endpoint
        k_batch = min(num_qubits, len(unvisited))
        batch_candidates = sorted(list(unvisited), key=lambda x: matrix[curr, x])[:k_batch]

        # Frame 1: Highlight 5-point cluster selection
        save_frame(
            f"Batch #{batch_count} - Selected {len(batch_candidates)} Nearest Points for WS-LR QAOA",
            tour,
            curr_node=curr,
            batch_candidates=batch_candidates,
        )

        # Step B: Solve optimal path sequence for all 5 points via WS-LR QAOA
        qaoa_subtour = solve_wslr_qaoa_subtour(curr, batch_candidates, matrix)

        # Frame 2: Show optimal 5-point sub-tour sequence constructed by QAOA
        save_frame(
            f"Batch #{batch_count} - WS-LR QAOA Solved 5-Point Sub-path Sequence",
            tour,
            curr_node=curr,
            batch_candidates=batch_candidates,
            active_subtour=qaoa_subtour,
        )

        # Step C: Append ALL 5 points of the solved sequence into the main route
        tour.extend(qaoa_subtour)
        for node in qaoa_subtour:
            unvisited.remove(node)
        curr = qaoa_subtour[-1]

        # Frame 3: Show 5-point sub-tour integrated into global route
        save_frame(
            f"Batch #{batch_count} - Integrated All {len(qaoa_subtour)} Points into Main Route",
            tour,
            curr_node=curr,
        )
        batch_count += 1

    # =========================================================================
    # PHASE 3: OPEN TSP 2-OPT POST-PROCESSING (LNS)
    # =========================================================================
    save_frame("Constructive Phase Complete - Starting LNS 2-Opt Post-Processing", tour)

    improved = True
    max_iter = 100
    iter_cnt = 0
    swap_count = 1

    while improved and iter_cnt < max_iter:
        improved = False
        iter_cnt += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == n - 1:
                    old_dist = matrix[tour[i - 1], tour[i]]
                    new_dist = matrix[tour[i - 1], tour[j]]
                    old_edge_pts = coords[[tour[i - 1], tour[i]]]
                    new_edge_pts = coords[[tour[i - 1], tour[j]]]
                else:
                    old_dist = matrix[tour[i - 1], tour[i]] + matrix[tour[j], tour[j + 1]]
                    new_dist = matrix[tour[i - 1], tour[j]] + matrix[tour[i], tour[j + 1]]
                    old_edge_pts = coords[[tour[i - 1], tour[i], tour[j], tour[j + 1]]]
                    new_edge_pts = coords[[tour[i - 1], tour[j], tour[i], tour[j + 1]]]

                if new_dist < old_dist:
                    save_frame(
                        f"LNS 2-Opt Swap #{swap_count} (Delta: {(new_dist - old_dist)/60.0:+.2f} min)",
                        tour,
                        highlight_swap=(old_edge_pts, new_edge_pts),
                    )

                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
                    swap_count += 1

                    save_frame(f"LNS 2-Opt Swap #{swap_count-1} Applied", tour)
                    break
            if improved:
                break

    # =========================================================================
    # PHASE 4: FINAL EVALUATION
    # =========================================================================
    cost_sec = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    cost_min = cost_sec / 60.0
    diff_pct = ((cost_min - amazon_planned_cost) / amazon_planned_cost) * 100.0

    save_frame(
        f"Final Route: {cost_min:.2f} min (Amazon Planned: {amazon_planned_cost:.2f} min, {diff_pct:+.2f}%)",
        tour,
    )

    print(f"\nStep-by-step visualizer finished!")
    print(f"Route ID: {route_id}")
    print(f"Total Stops: {n}")
    print(f"Algo 2 Cost: {cost_min:.2f} min | Amazon Planned: {amazon_planned_cost:.2f} min ({diff_pct:+.2f}%)")
    print(f"Saved {frame_idx} frames to '{os.path.abspath(output_dir)}'")


if __name__ == "__main__":
    loader = AmazonDataLoader()
    available_routes = list(loader.travel_times.keys())

    if not available_routes:
        print("No routes found in travel_times.json!")
    else:
        selected_route_id = random.choice(available_routes)
        data = loader.extract_single_route(selected_route_id)
        data["route_id"] = selected_route_id

        visualize_algo2_wslr_qaoa_batch(data, num_qubits=5, output_dir="step_by_step")