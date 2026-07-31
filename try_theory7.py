#### Rolling Window WS-LR-QAOA with PCA Consensus Edge Extraction + LNS Post-Processing
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linprog

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# =====================================================================
# 1. DATA LOADER & COST EVALUATION
# =====================================================================
class AmazonDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.build_inputs = os.path.join(data_dir, "model_build_inputs")
        self.routes_meta = self.load("route_data.json")
        self.travel_times = self.load("travel_times.json")
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

# =====================================================================
# 2. 5-QUBIT WS-LR-QAOA LEAF SOLVER
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
# 3. PCA CONSENSUS EDGE EXTRACTION
# =====================================================================
def get_pca_consensus_edges(coords):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(coords)
    
    # Sort nodes along PC1 and PC2
    order_pc1 = np.argsort(proj[:, 0])
    order_pc2 = np.argsort(proj[:, 1])
    
    # Extract adjacent pairs (undirected)
    edges_pc1 = set()
    for i in range(len(order_pc1) - 1):
        u, v = order_pc1[i], order_pc1[i+1]
        edges_pc1.add(tuple(sorted((u, v))))
        
    edges_pc2 = set()
    for i in range(len(order_pc2) - 1):
        u, v = order_pc2[i], order_pc2[i+1]
        edges_pc2.add(tuple(sorted((u, v))))
        
    # Consensus: edges present in BOTH PC projections
    consensus_edges = edges_pc1.intersection(edges_pc2)
    return consensus_edges

# =====================================================================
# 4. ALGORITHMS: UNCONSTRAINED & PCA-CONSTRAINED ROLLING WINDOW QAOA
# =====================================================================
def solve_rolling_window_qaoa(cost_matrix, depot_idx=0, window_size=5):
    n = len(cost_matrix)
    visited = {depot_idx}
    route = [depot_idx]
    curr_node = depot_idx
    
    while len(visited) < n:
        unvisited = [idx for idx in range(n) if idx not in visited]
        
        if len(unvisited) <= window_size - 1:
            candidates = unvisited
        else:
            sorted_unvisited = sorted(unvisited, key=lambda u: cost_matrix[curr_node, u])
            candidates = sorted_unvisited[: window_size - 1]
            
        sub_nodes = [curr_node] + candidates
        sub_cost = cost_matrix[np.ix_(sub_nodes, sub_nodes)]
        
        local_order = run_ws_lr_qaoa_leaf(sub_cost)
        ordered_sub_nodes = [sub_nodes[i] for i in local_order]
        curr_pos = ordered_sub_nodes.index(curr_node)
        rotated_path = ordered_sub_nodes[curr_pos:] + ordered_sub_nodes[:curr_pos]
        
        for node in rotated_path[1:]:
            if node not in visited:
                route.append(node)
                visited.add(node)
                
        curr_node = route[-1]
        
    return route

def solve_pca_rolling_window_qaoa(cost_matrix, coords, depot_idx=0, window_size=5):
    """
    Executes Rolling Window QAOA while preserving consensus edges from PC1 & PC2 projections.
    """
    consensus_edges = get_pca_consensus_edges(coords)
    
    # Build adjacency mapping for consensus edges
    adj = {}
    for u, v in consensus_edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        
    n = len(cost_matrix)
    visited = {depot_idx}
    route = [depot_idx]
    curr_node = depot_idx
    
    while len(visited) < n:
        unvisited = [idx for idx in range(n) if idx not in visited]
        
        # Priority 1: Force visit consensus neighbors if unvisited
        forced_neighbors = [nbr for nbr in adj.get(curr_node, []) if nbr not in visited]
        
        if forced_neighbors:
            candidates = forced_neighbors
            remaining_slots = (window_size - 1) - len(candidates)
            if remaining_slots > 0:
                other_unvisited = [u for u in unvisited if u not in candidates]
                sorted_others = sorted(other_unvisited, key=lambda u: cost_matrix[curr_node, u])
                candidates.extend(sorted_others[:remaining_slots])
        else:
            if len(unvisited) <= window_size - 1:
                candidates = unvisited
            else:
                sorted_unvisited = sorted(unvisited, key=lambda u: cost_matrix[curr_node, u])
                candidates = sorted_unvisited[: window_size - 1]
                
        sub_nodes = [curr_node] + candidates
        sub_cost = cost_matrix[np.ix_(sub_nodes, sub_nodes)]
        
        local_order = run_ws_lr_qaoa_leaf(sub_cost)
        ordered_sub_nodes = [sub_nodes[i] for i in local_order]
        curr_pos = ordered_sub_nodes.index(curr_node)
        rotated_path = ordered_sub_nodes[curr_pos:] + ordered_sub_nodes[:curr_pos]
        
        for node in rotated_path[1:]:
            if node not in visited:
                route.append(node)
                visited.add(node)
                
        curr_node = route[-1]
        
    return route

# =====================================================================
# 5. POST-PROCESSING & EXECUTION PIPELINE
# =====================================================================
def apply_lns_postprocessing(route, cost_matrix, max_iters=300, destroy_size_range=(4, 6), max_no_improve=50):
    best_route = list(route)
    best_cost = compute_total_route_cost(best_route, cost_matrix)
    n = len(best_route)
    
    if n <= max(destroy_size_range):
        return best_route

    no_improve_count = 0
    for _ in range(max_iters):
        if no_improve_count >= max_no_improve:
            break
            
        k = random.randint(destroy_size_range[0], destroy_size_range[1])
        start_idx = random.randint(0, n - k)
        
        destroyed_nodes = best_route[start_idx : start_idx + k]
        remaining_route = best_route[:start_idx] + best_route[start_idx + k:]
        
        candidate_route = list(remaining_route)
        for node in destroyed_nodes:
            best_pos = None
            min_cost_incr = float('inf')
            m = len(candidate_route)
            for pos in range(m):
                prev_node = candidate_route[pos - 1]
                next_node = candidate_route[pos]
                incr = cost_matrix[prev_node, node] + cost_matrix[node, next_node] - cost_matrix[prev_node, next_node]
                if incr < min_cost_incr:
                    min_cost_incr = incr
                    best_pos = pos
            candidate_route.insert(best_pos, node)
            
        candidate_cost = compute_total_route_cost(candidate_route, cost_matrix)
        if candidate_cost < best_cost - 1e-4:
            best_cost = candidate_cost
            best_route = candidate_route
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    return best_route

if __name__ == "__main__":
    DATA_DIR = "./almrrc2021-data-training"
    loader = AmazonDataLoader(DATA_DIR)
    
    available_routes = list(loader.travel_times.keys())
    target_route_id = available_routes[1] if len(available_routes) > 1 else available_routes[0]
        
    nodes, node_to_idx, cost_matrix, coords, actual_seq, depot_idx = loader.extract_single_route(target_route_id)
    num_stops = len(nodes)
    
    # 1. Planned Amazon Ground Truth
    amazon_cost = compute_total_route_cost(actual_seq, cost_matrix)
    
    # 2. Original Rolling Window WS-LR-QAOA (+ LNS)
    roll_raw = solve_rolling_window_qaoa(cost_matrix, depot_idx=depot_idx, window_size=5)
    roll_lns = apply_lns_postprocessing(roll_raw, cost_matrix, max_iters=300)
    roll_cost = compute_total_route_cost(roll_lns, cost_matrix)
    
    # 3. PCA Consensus Edge Intersection + Rolling Window WS-LR-QAOA (+ LNS)
    pca_roll_raw = solve_pca_rolling_window_qaoa(cost_matrix, coords, depot_idx=depot_idx, window_size=5)
    pca_roll_lns = apply_lns_postprocessing(pca_roll_raw, cost_matrix, max_iters=300)
    pca_roll_cost = compute_total_route_cost(pca_roll_lns, cost_matrix)
    
    # Output Table
    print("=" * 85)
    print(f"BENCHMARK COMPARISON FOR ROUTE: {target_route_id[:18]}...")
    print(f"Total Route Stops: {num_stops}")
    print("=" * 85)
    print(f"{'Algorithm / Architecture':<50} | {'Duration (min)':<15} | {'Gap vs Amazon (%)':<15}")
    print("-" * 85)
    print(f"{'Amazon Target Sequence (Ground Truth)':<50} | {amazon_cost:<15.2f} | {'0.00%':<15}")
    print(f"{'Original Rolling Window WS-LR-QAOA (+ LNS)':<50} | {roll_cost:<15.2f} | {((roll_cost - amazon_cost)/amazon_cost)*100:+.2f}%")
    print(f"{'PCA Intersection + Rolling Window QAOA (+ LNS)':<50} | {pca_roll_cost:<15.2f} | {((pca_roll_cost - amazon_cost)/amazon_cost)*100:+.2f}%")
    print("=" * 85)

    # Plot Routes
    non_depot_mask = np.ones(num_stops, dtype=bool)
    non_depot_mask[depot_idx] = False
    delivery_coords = coords[non_depot_mask]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    
    routes_to_plot = [
        ("Amazon Target", actual_seq, amazon_cost, '#1E88E5'),
        ("Original Rolling Window QAOA + LNS", roll_lns, roll_cost, '#2E7D32'),
        ("PCA Intersection + Rolling Window + LNS", pca_roll_lns, pca_roll_cost, '#D9381E')
    ]
    
    for ax, (title, seq, cost, color) in zip(axes, routes_to_plot):
        filt_seq = [i for i in seq if i != depot_idx]
        pts = coords[filt_seq]
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=1.2, alpha=0.85)
        ax.scatter(delivery_coords[:, 1], delivery_coords[:, 0], c='black', s=18)
        ax.set_title(f"{title}\nDuration: {cost:.1f} min", fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.grid(True, linestyle=':', alpha=0.6)
        
    axes[0].set_ylabel("Latitude")
    plt.suptitle("Amazon Target vs Original Rolling Window QAOA vs PCA Intersection Rolling Window QAOA", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()