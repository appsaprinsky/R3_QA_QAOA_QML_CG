class PricingSubproblem:
    def __init__(self, nodes, cost_matrix):
        self.nodes = nodes
        self.cost_matrix = cost_matrix
        self.n = len(nodes)
        self.depot_idx = 0

    def detect_negative_cycle_and_path(self, duals):
        """
        Uses Bellman-Ford algorithm to find negative reduced cost paths/cycles 
        by treating reduced costs as edge weights.
        Reduced Cost(u, v) = Cost(u, v) - Dual(v)
        """
        n = self.n
        dist = {i: float('inf') for i in range(n)}
        predecessor = {i: None for i in range(n)}
        
        # Start from the depot
        dist[self.depot_idx] = 0.0

        # Relax edges up to n-1 times
        for _ in range(n - 1):
            updated = False
            for u in range(n):
                if dist[u] == float('inf'):
                    continue
                for v in range(n):
                    if u == v:
                        continue
                    # Base cost + dual adjustment for pricing
                    base_cost = self.cost_matrix[u, v]
                    dual_val = duals.get(self.nodes[v], 0.0) if v != self.depot_idx else 0.0
                    reduced_cost = base_cost - dual_val

                    if dist[u] + reduced_cost < dist[v]:
                        dist[v] = dist[u] + reduced_cost
                        predecessor[v] = u
                        updated = True
            if not updated:
                break

        # Check for negative cycles (additional relaxation pass)
        negative_cycle_node = None
        for u in range(n):
            if dist[u] == float('inf'):
                continue
            for v in range(n):
                if u == v:
                    continue
                base_cost = self.cost_matrix[u, v]
                dual_val = duals.get(self.nodes[v], 0.0) if v != self.depot_idx else 0.0
                reduced_cost = base_cost - dual_val

                if dist[u] + reduced_cost < dist[v]:
                    negative_cycle_node = v
                    predecessor[v] = u
                    break
            if negative_cycle_node is not None:
                break

        if negative_cycle_node is not None:
            # Trace back cycle nodes
            cycle = []
            curr = negative_cycle_node
            for _ in range(n):
                curr = predecessor[curr]
            
            start_node = curr
            node_in_cycle = curr
            while True:
                cycle.append(node_in_cycle)
                node_in_cycle = predecessor[node_in_cycle]
                if node_in_cycle == start_node or node_in_cycle is None:
                    cycle.append(start_node)
                    break
            cycle.reverse()
            return cycle

        return None

    def solve(self, duals):
        """
        Solves the pricing subproblem via Bellman-Ford shortest path / negative cycle detection.
        """
        cycle_indices = self.detect_negative_cycle_and_path(duals)
        
        if not cycle_indices:
            return [], 0.0

        new_route_nodes = [self.nodes[idx] for idx in cycle_indices]
        
        # Calculate total true cost of the path
        route_cost = 0.0
        for i in range(len(cycle_indices) - 1):
            u = cycle_indices[i]
            v = cycle_indices[i+1]
            route_cost += self.cost_matrix[u, v]

        return new_route_nodes, route_cost