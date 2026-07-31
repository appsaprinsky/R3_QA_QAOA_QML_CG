import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

def visualize_pca_shortest_connection_graph(coords):
    """
    1. Projects points onto PC1 and PC2.
    2. Builds candidate edges from 1D sequential orderings.
    3. For each node, selects ONLY its single shortest 2D distance connection 
       among its PCA projection options.
    """
    n_points = len(coords)
    
    # 1. PCA Projections
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(coords)
    
    order_pc1 = np.argsort(coords_pca[:, 0])
    order_pc2 = np.argsort(coords_pca[:, 1])
    
    # Map node index to its neighbors in PC1 and PC2 1D sequences
    pca_candidate_edges = {i: set() for i in range(n_points)}
    
    for seq in [order_pc1, order_pc2]:
        for idx in range(len(seq) - 1):
            u, v = seq[idx], seq[idx+1]
            pca_candidate_edges[u].add(v)
            pca_candidate_edges[v].add(u)
            
    # 2. For each node, pick the SINGLE shortest edge out of its PCA candidate set
    selected_edges = set()
    
    for u in range(n_points):
        candidates = list(pca_candidate_edges[u])
        if not candidates:
            continue
            
        # Compute 2D distances to candidate neighbors from PCA
        dists = [euclidean(coords[u], coords[v]) for v in candidates]
        best_neighbor = candidates[np.argmin(dists)]
        
        # Store edge as sorted tuple to prevent duplicate reverse-drawing
        edge = tuple(sorted((u, best_neighbor)))
        selected_edges.add(edge)

    # 3. Visualization
    plt.figure(figsize=(10, 8))
    
    # Draw selected PCA shortest edges
    for u, v in selected_edges:
        plt.plot(
            [coords[u, 0], coords[v, 0]], 
            [coords[u, 1], coords[v, 1]], 
            color='#D9381E', linestyle='-', linewidth=1.4, alpha=0.85, zorder=2
        )
        
    # Plot Stops and Depot
    plt.scatter(coords[:, 0], coords[:, 1], c='black', s=35, alpha=0.7, zorder=3, label='Stops')
    plt.scatter(coords[0, 0], coords[0, 1], c='#00E676', s=140, edgecolors='black', linewidth=1.5, zorder=4, label='Depot / Start')

    plt.title(
        f"PCA-Derived Single Shortest Edge Graph ({n_points} Nodes, {len(selected_edges)} Unique Edges)\n"
        f"Filter: Minimum 2D Distance Edge Selected Per Node from PC1/PC2 Projections", 
        fontsize=11, fontweight='bold', pad=12
    )
    plt.xlabel("X Coordinate", fontsize=11)
    plt.ylabel("Y Coordinate", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()

# =====================================================================
# EXAMPLE RUN WITH SYNTHETIC ROUTE DATA
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    
    t = np.random.uniform(0, 50, 150)
    x = t * 1.5 + np.random.normal(0, 4, 150)
    y = t * 0.5 + np.random.normal(0, 8, 150)
    sample_amazon_route = np.column_stack((x, y))
    
    visualize_pca_shortest_connection_graph(sample_amazon_route)