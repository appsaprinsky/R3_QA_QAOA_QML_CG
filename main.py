import os
from data_loader import AmazonDataLoader
from subproblem import PricingSubproblem
from master_problem import MasterProblem

def run_column_generation_pipeline(data_dir, max_iterations=5):
    print("Loading Amazon VRP dataset...")
    loader = AmazonDataLoader(data_dir)
    
    if not loader.routes_meta:
        print("No route metadata found. Verify dataset path.")
        return

    sample_route_id = list(loader.routes_meta.keys())[0]
    print(f"Selected Sample Route ID: {sample_route_id}")

    nodes, node_to_idx, cost_matrix = loader.extract_single_route(sample_route_id)
    customers = nodes[1:]

    # Initialize Master Problem
    master = MasterProblem(customers)
    subproblem = PricingSubproblem(nodes, cost_matrix)

    # Initialize with trivial single-stop dummy columns
    for cust in customers:
        cost = cost_matrix[node_to_idx[nodes[0]], node_to_idx[cust]] + cost_matrix[node_to_idx[cust], node_to_idx[nodes[0]]]
        master.add_column([nodes[0], cust, nodes[0]], cost)

    print("Starting Column Generation iterations...")
    for iteration in range(max_iterations):
        # 1. Solve Master LP to get duals
        duals, obj_val = master.solve_lp_relaxation()
        if duals is None:
            print("Master problem failed to find optimal relaxation.")
            break
            
        print(f"Iteration {iteration+1}: Master LP Objective = {obj_val:.2f}")

        # 2. Solve Pricing Subproblem
        new_nodes, new_cost = subproblem.solve(duals)
        
        if len(new_nodes) <= 2:
            print("No more beneficial columns found.")
            break
            
        # Check duplicate column
        if any(col['nodes'] == new_nodes for col in master.columns):
            print("Generated column already exists. Stopping generation.")
            break

        print(f"  -> Added new column with cost: {new_cost:.2f} (Length: {len(new_nodes)-2} stops)")
        master.add_column(new_nodes, new_cost)

    print("\nExecuting Integer Fixing Heuristic...")
    final_routes = master.solve_integer_fixing()
    print(f"Optimization finished! Selected {len(final_routes)} optimized paths via PuLP.")
    
    for idx, r in enumerate(final_routes):
        print(f"  Route {idx+1}: {r['nodes']} (Cost: {r['cost']})")

if __name__ == "__main__":
    dataset_path = "./almrrc2021-data-training"
    if os.path.exists(dataset_path):
        run_column_generation_pipeline(dataset_path, max_iterations=5)
    else:
        print(f"Dataset directory '{dataset_path}' not found. Please verify download path.")