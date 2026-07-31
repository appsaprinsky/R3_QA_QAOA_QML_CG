import json
import os
import numpy as np

class AmazonDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.build_inputs = os.path.join(data_dir, "model_build_inputs")
        self.routes_meta = self.load("route_data.json")
        self.travel_times = self.load("travel_times.json")
        self.packages = self.load("package_data.json")

    def load(self, filename):
        path = os.path.join(self.build_inputs, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def extract_single_route(self, route_id):
        if route_id not in self.travel_times:
            raise ValueError(f"Route ID {route_id} not found.")
        
        cost_dict = self.travel_times[route_id]
        nodes = list(cost_dict.keys())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        cost_matrix = np.zeros((n, n))
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                cost_matrix[i, j] = cost_dict[u].get(v, 0.0)
                
        return nodes, node_to_idx, cost_matrix