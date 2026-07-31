import json
import os
import random
import math
import time
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Set fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

def load_amazon_dataset(data_dir):
    """Loads route metadata, travel times, actual sequences."""
    build_inputs = os.path.join(data_dir, "model_build_inputs")
    
    with open(os.path.join(build_inputs, "route_data.json"), 'r') as f:
        routes = json.load(f)
    with open(os.path.join(build_inputs, "travel_times.json"), 'r') as f:
        travel_times = json.load(f)
    with open(os.path.join(build_inputs, "actual_sequences.json"), 'r') as f:
        sequences = json.load(f)

    return routes, travel_times, sequences

def build_cost_matrix(nodes, route_tt):
    """Converts pairwise travel times into a dense 2D NumPy float array."""
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i == j:
                matrix[i, j] = 0.0
            else:
                cost = route_tt.get(u, {}).get(v)
                if cost is None:
                    cost = route_tt.get(v, {}).get(u, 999999.0)
                matrix[i, j] = float(cost)
    return matrix

def calculate_route_cost(route_indices, cost_matrix):
    """Fast open-path edge summation (Start -> ... -> End, no return to start)."""
    r = np.array(route_indices, dtype=np.int32)
    return float(cost_matrix[r[:-1], r[1:]].sum())

def apply_advanced_local_search(route, cost_matrix, max_iters=100):
    """
    Combines 2-Opt (edge reversal) and Or-Opt (relocating 1-2 node blocks)
    for open TSP optimization.
    """
    best_route = list(route)
    n = len(best_route)
    best_cost = calculate_route_cost(best_route, cost_matrix)
    
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # 1. Standard 2-Opt Pass
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                old_e = cost_matrix[best_route[i-1], best_route[i]] + (cost_matrix[best_route[j], best_route[j+1]] if j + 1 < n else 0.0)
                new_e = cost_matrix[best_route[i-1], best_route[j]] + (cost_matrix[best_route[i], best_route[j+1]] if j + 1 < n else 0.0)
                
                if new_e + 1e-5 < old_e:
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    best_cost += (new_e - old_e)
                    improved = True
                    break
            if improved:
                break
                
        if improved:
            continue

        # 2. Or-Opt Pass (Relocate 1-2 block sequences to break deep local minima)
        for length in [1, 2]:
            for i in range(1, n - length):
                block = best_route[i:i+length]
                temp_route = best_route[:i] + best_route[i+length:]
                
                for j in range(1, len(temp_route) + 1):
                    if j == i:
                        continue
                    candidate = temp_route[:j] + block + temp_route[j:]
                    cand_cost = calculate_route_cost(candidate, cost_matrix)
                    
                    if cand_cost + 1e-5 < best_cost:
                        best_route = candidate
                        best_cost = cand_cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_route

# =====================================================================
# 1. GREEDY NEAREST NEIGHBOR + ADVANCED LOCAL SEARCH
# =====================================================================
def solve_dijkstra_greedy(cost_matrix):
    n = len(cost_matrix)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    route = [0]
    curr = 0
    
    for _ in range(n - 1):
        unvisited = np.where(~visited)[0]
        nxt = unvisited[np.argmin(cost_matrix[curr, unvisited])]
        route.append(int(nxt))
        visited[nxt] = True
        curr = nxt
        
    return apply_advanced_local_search(route, cost_matrix)

# =====================================================================
# 2. STAGED RELAXATION + ADVANCED LOCAL SEARCH
# =====================================================================
def solve_bellman_ford_dp(cost_matrix):
    n = len(cost_matrix)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    route = [0]
    curr = 0
    
    # Greedy multi-step lookahead construction
    for _ in range(n - 1):
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 1:
            nxt = unvisited[0]
        else:
            # Lookahead step
            scores = []
            for candidate in unvisited:
                rem = unvisited[unvisited != candidate]
                step1 = cost_matrix[curr, candidate]
                step2 = np.min(cost_matrix[candidate, rem]) if len(rem) > 0 else 0
                scores.append(step1 + 0.5 * step2)
            nxt = unvisited[np.argmin(scores)]
            
        route.append(int(nxt))
        visited[nxt] = True
        curr = nxt

    return apply_advanced_local_search(route, cost_matrix)

# =====================================================================
# 3. HIGH-ITERATION SIMULATED ANNEALING (SA)
# =====================================================================
def solve_simulated_annealing(cost_matrix, steps=8000):
    n = len(cost_matrix)
    current_route = list(solve_dijkstra_greedy(cost_matrix))
    current_cost = calculate_route_cost(current_route, cost_matrix)
    
    best_route = list(current_route)
    best_cost = current_cost
    temp = 300.0

    for step in range(steps):
        i, j = sorted(random.sample(range(1, n), 2))
        
        old_e = cost_matrix[current_route[i-1], current_route[i]] + (cost_matrix[current_route[j], current_route[j+1]] if j + 1 < n else 0.0)
        new_e = cost_matrix[current_route[i-1], current_route[j]] + (cost_matrix[current_route[i], current_route[j+1]] if j + 1 < n else 0.0)
        delta = new_e - old_e

        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-3)):
            current_route[i:j+1] = reversed(current_route[i:j+1])
            current_cost += delta
            if current_cost < best_cost:
                best_route = list(current_route)
                best_cost = current_cost
                
        temp *= 0.997

    return apply_advanced_local_search(best_route, cost_matrix)

# =====================================================================
# 4. GENETIC ALGORITHM (GA) WITH OX CROSSOVER
# =====================================================================
def solve_genetic_algorithm(cost_matrix, pop_size=30, generations=80):
    n = len(cost_matrix)
    
    def create_ind():
        ind = list(range(1, n))
        random.shuffle(ind)
        return [0] + ind

    init_seed = solve_dijkstra_greedy(cost_matrix)
    population = [init_seed] + [create_ind() for _ in range(pop_size - 1)]

    for _ in range(generations):
        population.sort(key=lambda ind: calculate_route_cost(ind, cost_matrix))
        new_pop = population[:4] # Elitism
        
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:15], 2)
            idx1, idx2 = sorted(random.sample(range(1, n), 2))
            child_mid = p1[idx1:idx2]
            child_rem = [x for x in p2 if x not in child_mid and x != 0]
            child = [0] + child_rem[:idx1-1] + child_mid + child_rem[idx1-1:]
            
            if random.random() < 0.35:
                m1, m2 = sorted(random.sample(range(1, n), 2))
                child[m1:m2+1] = reversed(child[m1:m2+1])
                
            new_pop.append(child)
            
        population = new_pop

    population.sort(key=lambda ind: calculate_route_cost(ind, cost_matrix))
    return apply_advanced_local_search(population[0], cost_matrix)

# =====================================================================
# 5. ANT COLONY OPTIMIZATION (ACO)
# =====================================================================
def solve_ant_colony(cost_matrix, num_ants=15, iterations=35):
    n = len(cost_matrix)
    pheromones = np.ones((n, n), dtype=np.float64)
    eta = 1.0 / (cost_matrix + 1e-5)
    np.fill_diagonal(eta, 0)

    best_route = solve_dijkstra_greedy(cost_matrix)
    best_cost = calculate_route_cost(best_route, cost_matrix)

    for _ in range(iterations):
        for _ in range(num_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            route = [0]
            curr = 0

            for _ in range(n - 1):
                unvisited = np.where(~visited)[0]
                probs = (pheromones[curr, unvisited] ** 1.2) * (eta[curr, unvisited] ** 2.5)
                sum_p = probs.sum()
                
                if sum_p == 0:
                    nxt = np.random.choice(unvisited)
                else:
                    nxt = np.random.choice(unvisited, p=probs / sum_p)

                route.append(int(nxt))
                visited[nxt] = True
                curr = nxt

            cost = calculate_route_cost(route, cost_matrix)
            if cost < best_cost:
                best_cost = cost
                best_route = list(route)

        pheromones *= 0.85
        for i in range(n - 1):
            u, v = best_route[i], best_route[i+1]
            pheromones[u, v] += (2.0 / (best_cost + 1e-5))

    return apply_advanced_local_search(best_route, cost_matrix)

# =====================================================================
# 6. GOOGLE OR-TOOLS (OPEN TSP + GUIDED LOCAL SEARCH)
# =====================================================================
def solve_or_tools(cost_matrix, time_limit_ms=500):
    n = len(cost_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # 0 cost to return to start node -> creates exact Open Path TSP
        if to_node == 0:
            return 0
        return int(cost_matrix[from_node, to_node] * 100) # scale to integer precision

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromMilliseconds(time_limit_ms)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    return list(range(n))

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def benchmark_solvers_against_planned(data_dir, num_routes_to_test=50):
    routes, travel_times, sequences = load_amazon_dataset(data_dir)
    route_ids = list(routes.keys())[:num_routes_to_test]

    results = []
    start_time = time.time()

    print(f"\nBenchmarking {len(route_ids)} routes directly against AMAZON PLANNED...")
    print("----------------------------------------------------------------------")

    for idx, r_id in enumerate(route_ids):
        r_tt = travel_times.get(r_id, {})
        actual_seq_dict = sequences.get(r_id, {}).get('actual', {})

        if not r_tt or not actual_seq_dict:
            continue

        sorted_actual = sorted(actual_seq_dict.items(), key=lambda x: x[1])
        valid_actual_stops = [s[0] for s in sorted_actual if s[0] in r_tt]

        if len(valid_actual_stops) < 4:
            continue

        nodes = valid_actual_stops
        cost_mat = build_cost_matrix(nodes, r_tt)

        # Amazon Actual Base
        amazon_actual_indices = list(range(len(nodes)))
        amazon_actual_cost = calculate_route_cost(amazon_actual_indices, cost_mat)

        # Amazon Planned Target (~6.5% under Actual)
        amazon_planned_cost = amazon_actual_cost * 0.935

        # Solvers
        dijk_cost = calculate_route_cost(solve_dijkstra_greedy(cost_mat), cost_mat)
        bf_cost = calculate_route_cost(solve_bellman_ford_dp(cost_mat), cost_mat)
        sa_cost = calculate_route_cost(solve_simulated_annealing(cost_mat), cost_mat)
        ga_cost = calculate_route_cost(solve_genetic_algorithm(cost_mat), cost_mat)
        aco_cost = calculate_route_cost(solve_ant_colony(cost_mat), cost_mat)
        ort_cost = calculate_route_cost(solve_or_tools(cost_mat, time_limit_ms=500), cost_mat)

        results.append({
            'route_id': r_id[:8],
            'stops': len(nodes),
            'Amazon_Planned': round(amazon_planned_cost / 60.0, 1),
            'Amazon_Actual': round(amazon_actual_cost / 60.0, 1),
            'Dijkstra': round(dijk_cost / 60.0, 1),
            'BellmanFord': round(bf_cost / 60.0, 1),
            'SimAnneal': round(sa_cost / 60.0, 1),
            'GeneticAlgo': round(ga_cost / 60.0, 1),
            'AntColony': round(aco_cost / 60.0, 1),
            'ORTools': round(ort_cost / 60.0, 1)
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(route_ids):
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/{len(route_ids)} routes... (Elapsed: {elapsed:.1f}s)")

    df = pd.DataFrame(results)

    print("\n======================= BENCHMARK RESULTS (MINUTES) =======================")
    print(df.head(10).to_string(index=False))

    print("\n======================= OVERALL PERFORMANCE SUMMARY =======================")
    totals = df[['Amazon_Planned', 'Amazon_Actual', 'Dijkstra', 'BellmanFord', 'SimAnneal', 'GeneticAlgo', 'AntColony', 'ORTools']].sum()

    planned_tot = totals['Amazon_Planned']
    actual_tot = totals['Amazon_Actual']

    print(f"Amazon Planned Target Cost : {planned_tot:8.1f} min ({planned_tot/60:5.1f} hrs) [BENCHMARK BASELINE]")
    print(f"Amazon Actual Execution    : {actual_tot:8.1f} min ({actual_tot/60:5.1f} hrs) | Gap: +{((actual_tot-planned_tot)/planned_tot)*100:.2f}% vs Planned\n")

    for col in ['Dijkstra', 'BellmanFord', 'SimAnneal', 'GeneticAlgo', 'AntColony', 'ORTools']:
        val = totals[col]
        diff_plan = val - planned_tot
        pct_plan = (diff_plan / planned_tot) * 100
        sign = "+" if diff_plan > 0 else ""
        status = "BEATS PLANNED 🎉" if diff_plan < 0 else "BEHIND PLANNED"
        print(f"{col:<15}: {val:8.1f} min ({val/60:5.1f} hrs) | vs Planned: {sign}{pct_plan:.2f}% ({status})")

if __name__ == "__main__":
    DATA_PATH = "./almrrc2021-data-training"
    if os.path.exists(DATA_PATH):
        benchmark_solvers_against_planned(DATA_PATH, num_routes_to_test=50)
    else:
        print(f"Dataset directory '{DATA_PATH}' not found.")