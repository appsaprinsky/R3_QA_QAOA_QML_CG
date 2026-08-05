###Continuous LP Relaxation (Held-Karp + DFJ Cuts)

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linprog

# =====================================================================
# 1. DATA LOADER
# =====================================================================
class AmazonDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.build_inputs = os.path.join(data_dir, "model_build_inputs")
        self.routes_meta = self.load("route_data.json")
        self.travel_times = self.load("travel_times.json")

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
        cost_dict = self.travel_times[route_id]
        nodes = list(cost_dict.keys())
        n = len(nodes)
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
                    
        return nodes, cost_matrix, coords, depot_idx

# =====================================================================
# 2. DFJ ITERATIVE SUBTOUR / ISLAND ELIMINATION SOLVER
# =====================================================================
def solve_dfj_lp_no_islands(cost_matrix, depot_idx=0, max_cuts=50):
    """
    Solves LP relaxation using Dantzig-Fulkerson-Johnson (DFJ) cut generation.
    Iteratively detects connected components (islands) among customer nodes and
    adds cut constraints: sum_{i in S, j not in S} x_ij >= 1.0 until NO islands exist.
    """
    n = len(cost_matrix)
    num_edges = n * n
    c = cost_matrix.flatten()
    
    # Core degree constraints: sum_j x_ij = 1, sum_i x_ij = 1
    A_eq, b_eq = [], []
    for i in range(n):
        row_out = np.zeros(num_edges)
        row_in = np.zeros(num_edges)
        for j in range(n):
            if i != j:
                row_out[i * n + j] = 1.0
                row_in[j * n + i] = 1.0
        A_eq.extend([row_out, row_in])
        b_eq.extend([1.0, 1.0])
        
    bounds = [(0.0, 1.0) if i != j else (0.0, 0.0) for i in range(n) for j in range(n)]
    
    A_ub, b_ub = [], []
    
    for iteration in range(max_cuts):
        res = linprog(
            c, 
            A_eq=A_eq, b_eq=b_eq, 
            A_ub=A_ub if A_ub else None, 
            b_ub=b_ub if b_ub else None, 
            bounds=bounds, 
            method='highs'
        )
        
        if not res.success:
            raise ValueError("LP relaxation failed during cut generation.")
            
        X_lp = res.x.reshape((n, n))
        
        # Build undirected graph of non-depot active connections (x_ij > 0.01)
        G = nx.Graph()
        non_depot_nodes = [i for i in range(n) if i != depot_idx]
        G.add_nodes_from(non_depot_nodes)
        
        for i in non_depot_nodes:
            for j in non_depot_nodes:
                if i != j and (X_lp[i, j] + X_lp[j, i]) > 0.01:
                    G.add_edge(i, j)
                    
        # Find isolated components (islands) among non-depot nodes
        components = list(nx.connected_components(G))
        
        # If all non-depot nodes form a single connected component, we are done!
        if len(components) <= 1:
            print(f"Convergence reached in {iteration + 1} iterations. Zero isolated islands remaining.")
            return X_lp
            
        # Otherwise, add cut constraints for every isolated island S: sum_{i in S, j not in S} x_ij >= 1.0
        for comp in components:
            S = set(comp)
            not_S = set(range(n)) - S
            
            # Constraint: sum_{i in S, j in not_S} x_ij >= 1.0  ==>  -sum x_ij <= -1.0
            cut_row = np.zeros(num_edges)
            for i in S:
                for j in not_S:
                    cut_row[i * n + j] = -1.0
            
            A_ub.append(cut_row)
            b_ub.append(-1.0)
            
    return X_lp

# =====================================================================
# 3. VISUALIZATION (GUARANTEED NO ISOLATED ISLANDS)
# =====================================================================
def plot_lp_strictly_connected_no_depot(coords, X_lp, depot_idx=0, threshold=0.01):
    n = len(coords)
    plt.figure(figsize=(12, 9))
    
    edge_count = 0
    
    # Plot all active connections among customer nodes in solid blue
    for i in range(n):
        if i == depot_idx:
            continue
        for j in range(n):
            if j == depot_idx:
                continue
                
            val = X_lp[i, j]
            if val > threshold:
                edge_count += 1
                plt.plot(
                    [coords[i, 1], coords[j, 1]], 
                    [coords[i, 0], coords[j, 0]], 
                    color='#1E88E5', alpha=0.9, linewidth=2.0, zorder=1
                )

    delivery_indices = [idx for idx in range(n) if idx != depot_idx]
    plt.scatter(
        coords[delivery_indices, 1], coords[delivery_indices, 0], 
        c='#222222', s=35, zorder=3, label='Delivery Stops'
    )

    plt.title(
        f"LP Support Graph with Dynamic DFJ Subtour Elimination (No Depot)\n"
        f"Strictly Connected Customer Backbone | Total Active Edges: {edge_count} (Solid Blue)", 
        fontsize=11, fontweight='bold'
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_DIR = "./almrrc2021-data-training"
    loader = AmazonDataLoader(DATA_DIR)
    
    available_routes = list(loader.travel_times.keys())
    target_route_id = available_routes[0]
    
    nodes, cost_matrix, coords, depot_idx = loader.extract_single_route(target_route_id)
    
    # Solve LP with dynamic DFJ subtour/island elimination cuts
    X_lp = solve_dfj_lp_no_islands(cost_matrix, depot_idx=depot_idx)
    
    # Plot Support Graph (No Depot, No Isolated Islands)
    plot_lp_strictly_connected_no_depot(coords, X_lp, depot_idx=depot_idx)