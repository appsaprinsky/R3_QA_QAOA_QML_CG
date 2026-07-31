import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linprog

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# =====================================================================
# 1. AMAZON DATA LOADER
# =====================================================================
class AmazonDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.build_inputs = os.path.join(data_dir, "model_build_inputs")
        self.routes_meta = self.load("route_data.json")
        self.travel_times = self.load("travel_times.json")
        self.packages = self.load("package_data.json")
        self.actual_sequences = self.load("actual_sequences.json")

    def load(self, filename):
        path = os.path.join(self.build_inputs, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        alt_path = os.path.join(self.data_dir, filename)
        if os.path.exists(alt_path):
            with open(alt_path, 'r') as f:
                return json.load(f)
        return {}

    def extract_single_route(self, route_id):
        if route_id not in self.travel_times:
            raise ValueError(f"Route ID {route_id} not found in travel_times.json.")
        
        cost_dict = self.travel_times[route_id]
        nodes = list(cost_dict.keys())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        cost_matrix = np.zeros((n, n))
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                cost_matrix[i, j] = cost_dict[u].get(v, 0.0)
                
        route_meta = self.routes_meta.get(route_id, {})
        stops_meta = route_meta.get("stops", {})
        coords = np.zeros((n, 2))
        
        depot_idx = 0
        for i, node in enumerate(nodes):
            if node in stops_meta:
                coords[i] = [stops_meta[node]["lat"], stops_meta[node]["lng"]]
                if stops_meta[node].get("type") == "Station":
                    depot_idx = i
            else:
                coords[i] = [0.0, 0.0]
                
        actual_seq = []
        if route_id in self.actual_sequences:
            seq_dict = self.actual_sequences[route_id].get("actual", {})
            if seq_dict:
                first_key = next(iter(seq_dict.keys()))
                if first_key in node_to_idx:
                    sorted_stops = sorted(seq_dict.items(), key=lambda x: x[1])
                    actual_seq = [node_to_idx[item[0]] for item in sorted_stops if item[0] in node_to_idx]
                else:
                    actual_seq = [node_to_idx[node_id] for node_id in seq_dict.values() if node_id in node_to_idx]

        if not actual_seq:
            actual_seq = list(range(n))

        return nodes, node_to_idx, cost_matrix, coords, actual_seq, depot_idx

# =====================================================================
# 2. EVALUATOR & CLASSICAL SOLVERS
# =====================================================================
def compute_total_route_cost(route_indices, cost_matrix):
    if not route_indices:
        return 0.0
    n = len(route_indices)
    total_cost = 0.0
    for idx in range(n - 1):
        u, v = route_indices[idx], route_indices[idx+1]
        total_cost += cost_matrix[u, v]
    total_cost += cost_matrix[route_indices[-1], route_indices[0]]
    return total_cost / 60.0 if total_cost > 1000 else total_cost

def run_dijkstra_greedy_leaf(cost_sub):
    """Greedy/Dijkstra shortest path for a local sub-problem (e.g. <= 5 nodes)."""
    n = len(cost_sub)
    if n <= 1: return [0]
    visited = {0}
    route = [0]
    curr = 0
    while len(route) < n:
        unvisited = [k for k in range(n) if k not in visited]
        # Pick edge with minimum travel_time
        nxt = unvisited[np.argmin([cost_sub[curr, uv] for uv in unvisited])]
        route.append(nxt)
        visited.add(nxt)
        curr = nxt
    return route

# =====================================================================
# 3. 5-QUBIT WS-LR-QAOA SOLVER (LEAF PROBLEM)
# =====================================================================
def solve_held_karp_lp(cost_sub):
    n = len(cost_sub)
    if n <= 1:
        return np.ones((n, n)), cost_sub
    
    c = cost_sub.flatten()
    num_edges = n * n
    A_eq, b_eq = [], []
    
    for i in range(n):
        row_out, row_in = np.zeros(num_edges), np.zeros(num_edges)
        for j in range(n):
            if i != j:
                row_out[i * n + j] = 1.0
                row_in[j * n + i] = 1.0
        A_eq.extend([row_out, row_in])
        b_eq.extend([1.0, 1.0])
        
    bounds = [(0.0, 1.0) if i != j else (0.0, 0.0) for i in range(n) for j in range(n)]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        return res.x.reshape((n, n)), cost_sub
    else:
        inv = 1.0 / (cost_sub + 1e-3)
        np.fill_diagonal(inv, 0)
        return inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-6), cost_sub

def run_ws_lr_qaoa_leaf(cost_sub):
    n = len(cost_sub)
    if n == 1: return [0]
    if n == 2: return [0, 1]
        
    lp_weights, cost_mat = solve_held_karp_lp(cost_sub)
    qc = QuantumCircuit(n, n)
    
    for i in range(n):
        avg_w = np.mean(lp_weights[i])
        qc.ry(2.0 * np.arcsin(np.sqrt(np.clip(avg_w, 0.01, 0.99))), i)
        
    for i in range(n - 1):
        qc.rzz(0.7, i, i + 1)
    qc.rzz(0.7, n - 1, 0)
    for i in range(n):
        qc.rx(0.4, i)
    qc.measure(range(n), range(n))
    
    backend = AerSimulator()
    compiled_qc = transpile(qc, backend)
    result = backend.run(compiled_qc, shots=300).result()
    counts = result.get_counts()
    
    bitstring = max(counts, key=counts.get)
    visited = {0}
    route = [0]
    curr = 0
    while len(route) < n:
        unvisited = [k for k in range(n) if k not in visited]
        scores = []
        for uv in unvisited:
            bit_val = int(bitstring[uv % len(bitstring)])
            score = (lp_weights[curr, uv] + 1e-3) * (1.6 if bit_val == 1 else 0.8) / (cost_mat[curr, uv] + 1e-3)
            scores.append(score)
        nxt = unvisited[np.argmax(scores)]
        route.append(nxt)
        visited.add(nxt)
        curr = nxt
    return route

# =====================================================================
# 4. RECURSIVE DECOMPOSITION (GENERIC LEAF SOLVER SWITCH)
# =====================================================================
def solve_hierarchical_route(indices, cost_matrix, coords, solver_type="QAOA", max_points=5):
    """
    Decomposes routes into clusters of size <= max_points (5).
    Uses either QAOA or Dijkstra for solving the 5-point leaf sub-problems.
    """
    n_pts = len(indices)
    if n_pts <= max_points:
        sub_cost = cost_matrix[np.ix_(indices, indices)]
        if solver_type == "QAOA":
            local_order = run_ws_lr_qaoa_leaf(sub_cost)
        else:
            local_order = run_dijkstra_greedy_leaf(sub_cost)
        return [indices[i] for i in local_order]
        
    k = min(max_points, n_pts)
    sub_coords = coords[indices]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(sub_coords)
    
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    centroid_costs = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                centroid_costs[i, j] = np.linalg.norm(centroids[i] - centroids[j]) * 100.0
                
    if solver_type == "QAOA":
        macro_order = run_ws_lr_qaoa_leaf(centroid_costs)
    else:
        macro_order = run_dijkstra_greedy_leaf(centroid_costs)
    
    cluster_routes = []
    for c_idx in macro_order:
        cluster_members = [indices[i] for i in range(n_pts) if cluster_labels[i] == c_idx]
        if len(cluster_members) > 0:
            c_route = solve_hierarchical_route(cluster_members, cost_matrix, coords, solver_type=solver_type, max_points=max_points)
            cluster_routes.append(c_route)
            
    stitched_route = []
    for c_route in cluster_routes:
        if not stitched_route:
            stitched_route.extend(c_route)
        else:
            prev_tail = stitched_route[-1]
            costs = [cost_matrix[prev_tail, p] for p in c_route]
            best_entry_idx = int(np.argmin(costs))
            rotated_c_route = c_route[best_entry_idx:] + c_route[:best_entry_idx]
            stitched_route.extend(rotated_c_route)
            
    return stitched_route

# =====================================================================
# 5. EXECUTION & COMPARISON PIPELINE
# =====================================================================
if __name__ == "__main__":
    DATA_DIR = "./almrrc2021-data-training"
    loader = AmazonDataLoader(DATA_DIR)
    
    available_routes = list(loader.travel_times.keys())
    target_route_id = available_routes[0]
    
    nodes, node_to_idx, cost_matrix, coords, actual_seq, depot_idx = loader.extract_single_route(target_route_id)
    num_stops = len(nodes)
    
    print("=" * 65)
    print(f"BENCHMARK RESULTS FOR AMAZON ROUTE: {target_route_id[:18]}...")
    print(f"Total Route Stops: {num_stops} (5-node window decomposition)")
    print("=" * 65)
    
    # 1. Amazon Planned/Actual Ground Truth
    amazon_cost = compute_total_route_cost(actual_seq, cost_matrix)
    print(f"1. Amazon Target Sequence  : {amazon_cost:.2f} min")
    
    # 2. 5-Node Constrained Local Dijkstra
    dijkstra_seq = solve_hierarchical_route(list(range(num_stops)), cost_matrix, coords, solver_type="DIJKSTRA", max_points=5)
    dijkstra_cost = compute_total_route_cost(dijkstra_seq, cost_matrix)
    d_gap = ((dijkstra_cost - amazon_cost) / amazon_cost) * 100
    print(f"2. 5-Point Local Dijkstra  : {dijkstra_cost:.2f} min | Gap vs Amazon: {d_gap:+.2f}%")
    
    # 3. 5-Qubit WS-LR-QAOA
    qaoa_seq = solve_hierarchical_route(list(range(num_stops)), cost_matrix, coords, solver_type="QAOA", max_points=5)
    qaoa_cost = compute_total_route_cost(qaoa_seq, cost_matrix)
    q_gap = ((qaoa_cost - amazon_cost) / amazon_cost) * 100
    print(f"3. 5-Qubit WS-LR-QAOA      : {qaoa_cost:.2f} min | Gap vs Amazon: {q_gap:+.2f}%")
    
    # Comparison between QAOA and Dijkstra
    qd_gap = ((qaoa_cost - dijkstra_cost) / dijkstra_cost) * 100
    print("-" * 65)
    print(f"QAOA vs Local Dijkstra Performance Delta: {qd_gap:+.2f}%")
    print("=" * 65)

    # Visualizing 5-Point Dijkstra vs 5-Qubit QAOA (Excluding Depot)
    filtered_dijkstra_seq = [idx for idx in dijkstra_seq if idx != depot_idx]
    filtered_qaoa_seq = [idx for idx in qaoa_seq if idx != depot_idx]
    
    non_depot_mask = np.ones(num_stops, dtype=bool)
    non_depot_mask[depot_idx] = False
    delivery_coords = coords[non_depot_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    # Plot Dijkstra
    pts_d = coords[filtered_dijkstra_seq]
    axes[0].plot(pts_d[:, 1], pts_d[:, 0], color='#2E7D32', linewidth=1.2, alpha=0.85, label='Dijkstra Route')
    axes[0].scatter(delivery_coords[:, 1], delivery_coords[:, 0], c='black', s=25, zorder=3, label='Stops')
    axes[0].scatter(pts_d[0, 1], pts_d[0, 0], c='#00E676', s=100, edgecolors='k', zorder=4, label='First Stop')
    axes[0].set_title(f"5-Point Hierarchical Dijkstra\nDuration: {dijkstra_cost:.1f} min (Gap vs Amazon: {d_gap:+.1f}%)", fontweight='bold')
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()
    
    # Plot QAOA
    pts_q = coords[filtered_qaoa_seq]
    axes[1].plot(pts_q[:, 1], pts_q[:, 0], color='#D9381E', linewidth=1.2, alpha=0.85, label='QAOA Route')
    axes[1].scatter(delivery_coords[:, 1], delivery_coords[:, 0], c='black', s=25, zorder=3, label='Stops')
    axes[1].scatter(pts_q[0, 1], pts_q[0, 0], c='#00E676', s=100, edgecolors='k', zorder=4, label='First Stop')
    axes[1].set_title(f"5-Qubit Hierarchical WS-LR-QAOA\nDuration: {qaoa_cost:.1f} min (Gap vs Amazon: {q_gap:+.1f}%)", fontweight='bold')
    axes[1].set_xlabel("Longitude")
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()
    
    plt.suptitle(f"Amazon LMRRC Benchmark Comparison — Route {target_route_id[:18]}...", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()