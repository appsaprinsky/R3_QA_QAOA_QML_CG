"""
Data Loader for Amazon Last Mile Routing Dataset.
Parses Amazon PLANNED sequences from route_data.json and travel_times.json.
"""

import json
import os
import numpy as np


class AmazonDataLoader:

    def __init__(self, data_dir: str = "./almrrc2021-data-training"):
        self.data_dir = data_dir
        self.build_inputs = os.path.join(data_dir, "model_build_inputs")

        self.routes_meta = self.load("route_data.json")
        self.travel_times = self.load("travel_times.json")
        self.actual_sequences = self.load("actual_sequences.json")

    def load(self, filename):
        path = os.path.join(self.build_inputs, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        alt_path = os.path.join(self.data_dir, filename)
        if os.path.exists(alt_path):
            with open(alt_path, "r") as f:
                return json.load(f)
        return {}

    def extract_single_route(self, route_id):
        if route_id not in self.travel_times:
            raise ValueError(
                f"Route ID {route_id} not found in travel_times.json"
            )

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

        # --- Extract Amazon PLANNED Sequence ---
        planned_seq = []
        if route_id in self.routes_meta:
            stops_dict = route_meta.get("stops", {})
            seq_tuples = []
            for code, s_info in stops_dict.items():
                if code in node_to_idx:
                    pos = s_info.get(
                        "sequence_number",
                        s_info.get("planned_sequence", None),
                    )
                    if pos is not None:
                        seq_tuples.append((node_to_idx[code], pos))

            if seq_tuples:
                seq_tuples.sort(key=lambda x: x[1])
                planned_seq = [t[0] for t in seq_tuples]

        # Fallback to sequence dictionary if missing sequence_number
        if not planned_seq and route_id in self.actual_sequences:
            seq_dict = self.actual_sequences[route_id].get(
                "actual", self.actual_sequences[route_id].get("sequence", {})
            )
            if seq_dict:
                sorted_stops = sorted(seq_dict.items(), key=lambda x: x[1])
                planned_seq = [
                    node_to_idx[item[0]]
                    for item in sorted_stops
                    if item[0] in node_to_idx
                ]

        if not planned_seq:
            planned_seq = list(range(n))

        amazon_planned_cost = self.compute_route_cost(planned_seq, cost_matrix)

        return {
            "matrix": cost_matrix,
            "coords": coords,
            "depot_idx": depot_idx,
            "amazon_planned_sequence": planned_seq,
            "amazon_planned_cost": amazon_planned_cost,
            "n_nodes": n,
            "route_id": route_id,
        }

    def compute_route_cost(self, route_indices, cost_matrix):
        if not route_indices or len(route_indices) < 2:
            return 0.0
        n = len(route_indices)
        total_cost = 0.0
        for idx in range(n - 1):
            u, v = route_indices[idx], route_indices[idx + 1]
            total_cost += cost_matrix[u, v]
        total_cost += cost_matrix[route_indices[-1], route_indices[0]]
        return total_cost / 60.0 if total_cost > 1000 else total_cost