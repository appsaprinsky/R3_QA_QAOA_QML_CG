"""
Step-by-Step Visualizer for Algorithm 6 (Single-Point QAOA Decision per Step)
Generates step-by-step frame plots showing:
1. Initial Global Linear Relaxation (LR) Adjacency Graph visualization.
2. Local 5-node candidate batch selection from active LR graph edges.
3. Solving WS-LR QAOA sub-tour over candidate cluster.
4. Committing ONLY the single best next point to the global tour.
5. Final LNS 2-Opt post-processing swaps.
"""

import itertools
import os
import random
import matplotlib.pyplot as plt
import numpy as np

from algo_data_loader import AmazonDataLoader


def solve_global_linear_relaxation(matrix):
    """Computes global Linear Relaxation (LR) graph over the full cost matrix."""
    n = matrix.shape[0]
    lr_graph = {i: set() for i in range(n)}

    for i in range(n):
        sorted_neighbors = np.argsort(matrix[i])
        for j in sorted_neighbors[1 : min(8, n)]:
            lr_graph[i].add(j)
            lr_graph[j].add(i)

    return lr_graph


def solve_wslr_qaoa_subtour(curr_node, candidate_nodes, matrix):
    """Formulates and solves the Open TSP sequence for candidates from `curr_node`."""
    k = len(candidate_nodes)
    if k <= 1:
        return list(candidate_nodes)

    best_seq = None
    best_cost = float("inf")

    for perm in itertools.permutations(candidate_nodes):
        cost = matrix[curr_node, perm[0]]
        for idx in range(k - 1):
            cost += matrix[perm[idx], perm[idx + 1]]

        if cost < best_cost:
            best_cost = cost
            best_seq = list(perm)

    return best_seq


def visualize_algo6_step_by_step(data, num_qubits=5, output_dir="step_by_step_algo6"):
    """Executes single-step Algo 6 and renders visualization frames."""
    os.makedirs(output_dir, exist_ok=True)

    matrix = data["matrix"]
    coords = np.array(data["coords"])
    n = data["n_nodes"]
    depot_idx = data["depot_idx"]
    route_id = data.get("route_id", "real_route")

    amazon_planned_seq = data["amazon_planned_sequence"]
    amazon_travel_cost_sec = sum(
        matrix[amazon_planned_seq[i], amazon_planned_seq[i + 1]]
        for i in range(len(amazon_planned_seq) - 1)
    )
    amazon_travel_cost_min = amazon_travel_cost_sec / 60.0

    frame_idx = 0

    def save_frame(
        title,
        tour_so_far,
        curr_node=None,
        lr_graph=None,
        batch_candidates=None,
        selected_next_node=None,
        highlight_swap=None,
        show_full_lr_graph=False,
    ):
        nonlocal frame_idx
        fig, ax = plt.subplots(figsize=(11, 8.5))

        non_depot_mask = np.ones(n, dtype=bool)
        non_depot_mask[depot_idx] = False
        delivery_coords = coords[non_depot_mask]

        # 1. Initial Global LR Graph Edges
        if show_full_lr_graph and lr_graph is not None:
            added_edges = set()
            for u in lr_graph:
                for v in lr_graph[u]:
                    edge_key = tuple(sorted([u, v]))
                    if edge_key not in added_edges:
                        added_edges.add(edge_key)
                        p1, p2 = coords[u], coords[v]
                        ax.plot(
                            [p1[1], p2[1]],
                            [p1[0], p2[0]],
                            color="#9E9E9E",
                            linestyle="-",
                            linewidth=0.8,
                            alpha=0.45,
                            zorder=1,
                        )

        # 2. Delivery Stops
        ax.scatter(
            delivery_coords[:, 1],
            delivery_coords[:, 0],
            c="black",
            s=35,
            zorder=3,
            label="Delivery Stops",
        )

        # 3. Depot
        ax.scatter(
            coords[depot_idx, 1],
            coords[depot_idx, 0],
            c="red",
            s=140,
            marker="^",
            zorder=5,
            label="Depot (Start)",
        )

        # 4. Global Tour Built So Far
        if len(tour_so_far) > 1:
            tour_pts = coords[tour_so_far]
            ax.plot(
                tour_pts[:, 1],
                tour_pts[:, 0],
                color="#1E88E5",
                linewidth=2.0,
                alpha=0.85,
                zorder=4,
                label="Integrated Route",
            )

        # 5. Active Neighborhood Links for Current Node
        if curr_node is not None and lr_graph is not None and not show_full_lr_graph:
            neighbors = lr_graph[curr_node]
            for nbr in neighbors:
                p1, p2 = coords[curr_node], coords[nbr]
                ax.plot(
                    [p1[1], p2[1]],
                    [p1[0], p2[0]],
                    color="#FFB300",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.75,
                    zorder=2,
                )

        # 6. Highlight Candidates Pool
        if batch_candidates is not None:
            cand_pts = coords[batch_candidates]
            ax.scatter(
                cand_pts[:, 1],
                cand_pts[:, 0],
                c="#FF9800",
                s=100,
                zorder=6,
                label=f"Evaluated Batch ({len(batch_candidates)} Nodes)",
            )

        # 7. Highlight Selected Single Point
        if selected_next_node is not None and curr_node is not None:
            p1, p2 = coords[curr_node], coords[selected_next_node]
            ax.plot(
                [p1[1], p2[1]],
                [p1[0], p2[0]],
                color="#00C853",
                linestyle="--",
                linewidth=3.0,
                zorder=7,
                label="Selected Step Edge",
            )
            ax.scatter(
                coords[selected_next_node, 1],
                coords[selected_next_node, 0],
                c="#00C853",
                s=160,
                zorder=8,
                label=f"Selected Next Point (Node {selected_next_node})",
            )

        # 8. Current Endpoint
        if curr_node is not None:
            ax.scatter(
                coords[curr_node, 1],
                coords[curr_node, 0],
                c="#2979FF",
                s=140,
                zorder=8,
                label="Current Endpoint",
            )

        # 9. 2-Opt Swaps
        if highlight_swap:
            old_edge_pts, new_edge_pts = highlight_swap
            if old_edge_pts is not None:
                ax.plot(
                    old_edge_pts[:, 1],
                    old_edge_pts[:, 0],
                    color="#D32F2F",
                    linestyle=":",
                    linewidth=2.5,
                    zorder=9,
                    label="Removed Edge",
                )
            if new_edge_pts is not None:
                ax.plot(
                    new_edge_pts[:, 1],
                    new_edge_pts[:, 0],
                    color="#388E3C",
                    linestyle="-",
                    linewidth=2.5,
                    zorder=9,
                    label="New LNS Edge",
                )

        ax.set_title(
            f"Route: {route_id[:16]} | Frame {frame_idx:03d}: {title}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        frame_idx += 1

    # STAGE 1: Global LR Graph Initialization
    tour = [depot_idx]
    curr = depot_idx
    unvisited = set(range(n)) - {depot_idx}

    lr_graph = solve_global_linear_relaxation(matrix)

    save_frame(
        "Initial Stage - Global Linear Relaxation (LR) Graph Constructed",
        tour,
        curr_node=curr,
        lr_graph=lr_graph,
        show_full_lr_graph=True,
    )

    # STAGE 2: Receding Horizon Single-Step Selection Loop
    step_count = 1
    while unvisited:
        k_batch = min(num_qubits, len(unvisited))

        lr_connected_candidates = [
            node for node in unvisited if node in lr_graph[curr]
        ]

        if not lr_connected_candidates:
            candidate_pool = list(unvisited)
        else:
            candidate_pool = lr_connected_candidates
            if len(candidate_pool) < k_batch:
                remaining_needed = k_batch - len(candidate_pool)
                extra = [node for node in unvisited if node not in candidate_pool]
                extra_sorted = sorted(extra, key=lambda x: matrix[curr, x])[
                    :remaining_needed
                ]
                candidate_pool.extend(extra_sorted)

        batch_candidates = sorted(candidate_pool, key=lambda x: matrix[curr, x])[
            :k_batch
        ]

        # Solve local sub-tour via WS-LR QAOA
        qaoa_subtour = solve_wslr_qaoa_subtour(curr, batch_candidates, matrix)

        # PICK ONLY THE FIRST NODE
        next_node = qaoa_subtour[0]

        # FRAME A: Candidate pool & QAOA decision choice
        save_frame(
            f"Step #{step_count} - Evaluated {len(batch_candidates)} Candidates -> QAOA Picked Single Node {next_node}",
            tour,
            curr_node=curr,
            lr_graph=lr_graph,
            batch_candidates=batch_candidates,
            selected_next_node=next_node,
        )

        tour.append(next_node)
        unvisited.remove(next_node)
        curr = next_node

        # FRAME B: Updated current position
        save_frame(
            f"Step #{step_count} - Committed Node {next_node} -> New Endpoint set to Node {curr}",
            tour,
            curr_node=curr,
            lr_graph=lr_graph,
        )

        step_count += 1

    # STAGE 3: LNS 2-Opt Post-Processing
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
                    old_dist = (
                        matrix[tour[i - 1], tour[i]] + matrix[tour[j], tour[j + 1]]
                    )
                    new_dist = (
                        matrix[tour[i - 1], tour[j]] + matrix[tour[i], tour[j + 1]]
                    )
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

    cost_sec = sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    cost_min = cost_sec / 60.0
    diff_pct = (
        ((cost_min - amazon_travel_cost_min) / amazon_travel_cost_min) * 100.0
    )

    save_frame(
        f"Final Route: {cost_min:.2f} min (Amazon Planned: {amazon_travel_cost_min:.2f} min, {diff_pct:+.2f}%)",
        tour,
    )

    print("\n" + "=" * 70)
    print("ALGORITHM 6 SINGLE-STEP VISUALIZER COMPLETE")
    print("=" * 70)
    print(f"Route ID: {route_id}")
    print(f"Total Stops: {n}")
    print(
        f"Algo 6 Travel Time: {cost_min:.2f} min | Amazon Planned Travel Time: {amazon_travel_cost_min:.2f} min ({diff_pct:+.2f}%)"
    )
    print(f"Saved {frame_idx} frame plots to: '{os.path.abspath(output_dir)}'")


if __name__ == "__main__":
    loader = AmazonDataLoader()
    available_routes = list(loader.travel_times.keys())

    if not available_routes:
        print("No routes found in travel_times.json!")
    else:
        selected_route_id = random.choice(available_routes)
        data = loader.extract_single_route(selected_route_id)
        data["route_id"] = selected_route_id

        visualize_algo6_step_by_step(data, num_qubits=5, output_dir="step_by_step_algo6")