"""
Data Loader for Amazon Last Mile Routing Dataset.
Parses Amazon PLANNED sequences from route_data.json and travel_times.json.

--------------------------------------------------------------------------
FIX LOG (this revision)
--------------------------------------------------------------------------
* THE ACTUAL ROOT CAUSE of "the visualization is unreadable, points look
  too far apart" across MULTIPLE scripts (run_amazon_experiment.py,
  run_CG_experiment.py, run_experiment_ALL.py, and especially
  visualize_step_by_step_CG_REAL_DATA.py): a node missing from
  stops_meta kept its default coords of (0, 0) -- a real lat/lng in the
  Gulf of Guinea, nowhere near any real Amazon route. Every caller-side
  fallback (`if coords is None or np.all(coords == 0): use MDS`) only
  triggers when EVERY node is zero, not when a handful are -- so on any
  route where a few (not all) stops were missing metadata, those (0,0)
  points silently entered the coordinate array, stretched every plot's
  auto-computed axis limits from "the size of a delivery zone" to "the
  size of the Atlantic Ocean", and made the real ~100-250 point cluster
  collapse to a single invisible dot in the corner of every figure.
  A previous revision's FIX LOG already *described* handling this ("MDS
  fallback triggers if any required node is missing, not just all") but
  that was never actually wired up here -- only a warning was added; the
  reconstruction itself was still left to each caller's now-provably-
  insufficient np.all(coords==0) check. Fixed for real this time:
  extract_single_route() now reconstructs coords for the WHOLE route via
  classical MDS on the (always-complete) cost matrix whenever
  missing_coord_nodes is non-empty -- not just when every node is
  missing -- centralizing the fix at the single source every script
  already loads through, instead of depending on 3-4 separate caller-
  side checks staying in sync with what "degenerate" means.
--------------------------------------------------------------------------
* compute_route_cost(): previously computed a CLOSED-loop cost (added
  `cost_matrix[route_indices[-1], route_indices[0]]`, i.e. a return leg
  back to the start) and applied an ad hoc `/60.0 if total_cost > 1000`
  unit-conversion heuristic. Neither of those matches
  `compute_open_route_cost()` in run_amazon_experiment.py /
  compare_amazon_planned.py, which is what's actually used for every
  reported Amazon-vs-Hybrid comparison (open route, no return leg, no
  unit conversion). The two functions were never called on the same
  metric in the current scripts, so today's headline numbers are NOT
  affected -- but `amazon_planned_cost` (computed with the old, different
  formula) is still attached to every loaded route's data dict, and any
  future script/notebook that reads it directly instead of recomputing
  via compute_open_route_cost() would silently get a number on a
  different basis (closed-loop, possibly minutes instead of raw units).
  Fixed to match the open-route, no-conversion convention used
  everywhere else, so the field is safe to read directly from now on.
* extract_single_route(): added `validate_route()` and call it before
  returning, to check the two things that actually determine whether an
  Amazon-vs-Hybrid comparison is apples-to-apples: (1) does the planned
  sequence cover every node exactly once (some stops can silently be
  dropped from the planned sequence if they lack a sequence_number in
  the source data), and (2) does it start at the depot (the hybrid tour
  always does, since run_algo_hybrid_2_5() explicitly starts at
  depot_idx -- if the planned sequence doesn't, the two costs are not
  measuring the same trip). Logs a warning rather than raising, since a
  malformed route in a 10-route sample shouldn't crash the whole run,
  but you want to know if it happened.
* extract_single_route(): depot_idx now breaks on the first "Station"
  stop and warns if more than one is found, instead of silently taking
  whichever "Station" happens to be last in dict-iteration order.
--------------------------------------------------------------------------
"""

import json
import os
import warnings
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
        station_count = 0
        missing_coord_nodes = []
        for i, node in enumerate(nodes):
            if node in stops_meta:
                coords[i] = [stops_meta[node]["lat"], stops_meta[node]["lng"]]
                if stops_meta[node].get("type") == "Station":
                    station_count += 1
                    if station_count == 1:
                        depot_idx = i
            else:
                missing_coord_nodes.append(node)

        if station_count > 1:
            warnings.warn(
                f"[{route_id}] found {station_count} stops of type 'Station'; "
                f"using the first one (index {depot_idx}) as depot_idx. "
                f"Verify this is correct for this route."
            )

        if missing_coord_nodes:
            warnings.warn(
                f"[{route_id}] {len(missing_coord_nodes)}/{n} node(s) missing from "
                f"stops metadata: {missing_coord_nodes[:5]}"
                f"{'...' if len(missing_coord_nodes) > 5 else ''}. "
                f"Reconstructing coordinates for this route via MDS on the cost "
                f"matrix (see FIX LOG) instead of leaving them at the (0, 0) "
                f"default, which would otherwise distort every plot's axis scale."
            )
            coords = _reconstruct_coords_via_mds(cost_matrix, route_id, existing_coords=coords)

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

        planned_seq = self._validate_planned_sequence(route_id, planned_seq, n, depot_idx)

        # FIX: use the same cost convention as everywhere else that
        # actually reports a comparison (open route, no unit conversion).
        amazon_planned_cost = compute_open_route_cost(planned_seq, cost_matrix)

        return {
            "matrix": cost_matrix,
            "coords": coords,
            "depot_idx": depot_idx,
            "amazon_planned_sequence": planned_seq,
            "amazon_planned_cost": amazon_planned_cost,
            "n_nodes": n,
            "route_id": route_id,
            "missing_coord_nodes": missing_coord_nodes,
        }

    def _validate_planned_sequence(self, route_id, planned_seq, n, depot_idx):
        """
        Checks the two invariants that determine whether Amazon-planned vs.
        Hybrid tour costs are actually comparable:
          1. planned_seq visits every node in range(n) exactly once.
          2. planned_seq starts at depot_idx (run_algo_hybrid_2_5 always
             starts there, so if Amazon's sequence doesn't, the two costs
             describe different trips).
        Logs a warning and repairs rather than raising, so one malformed
        route doesn't kill a whole batch run -- but the repair is
        conservative (append missing nodes at the end, don't guess an
        order for them) and always logged so it's visible in output.
        """
        seq_set = set(planned_seq)
        expected_set = set(range(n))

        if len(planned_seq) != len(seq_set):
            warnings.warn(
                f"[{route_id}] planned_seq contains duplicate node indices; "
                f"deduplicating (first occurrence kept)."
            )
            seen = set()
            deduped = []
            for node in planned_seq:
                if node not in seen:
                    deduped.append(node)
                    seen.add(node)
            planned_seq = deduped
            seq_set = set(planned_seq)

        missing = expected_set - seq_set
        if missing:
            warnings.warn(
                f"[{route_id}] planned_seq is missing {len(missing)}/{n} node(s) "
                f"(likely due to missing sequence_number in source data): "
                f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}. "
                f"Appending them at the end so amazon_planned_cost covers the "
                f"same node set as the hybrid tour -- but this changes what "
                f"'Amazon Planned' means for this route; treat its comparison "
                f"with extra caution or exclude it from aggregate results."
            )
            planned_seq = planned_seq + sorted(missing)

        if planned_seq and planned_seq[0] != depot_idx:
            warnings.warn(
                f"[{route_id}] planned_seq does not start at depot_idx "
                f"({planned_seq[0]} != {depot_idx}). The Hybrid tour always "
                f"starts at the depot, so costs are not directly comparable "
                f"as-is. Moving depot_idx to the front to match convention."
            )
            planned_seq = [depot_idx] + [x for x in planned_seq if x != depot_idx]

        return planned_seq

    def compute_route_cost(self, route_indices, cost_matrix):
        """
        Kept for backward compatibility with any external caller, but now
        delegates to the same open-route convention used everywhere else
        instead of silently computing a different (closed-loop,
        conditionally-rescaled) number under the same-looking name.
        """
        return compute_open_route_cost(route_indices, cost_matrix)


def _reconstruct_coords_via_mds(cost_matrix, route_id, existing_coords):
    """
    Classical multidimensional scaling on the full pairwise cost matrix,
    producing a 2D layout for EVERY node in the route -- not just the
    ones that were missing metadata. This is deliberate: MDS needs one
    globally consistent embedding to be meaningful (there's no principled
    way to place only the missing points "relative to" the real ones
    without re-running the same global optimization), and since
    missing_coord_nodes is typically a small fraction of the route, the
    resulting layout for the well-known majority of nodes still tracks
    real relative geography closely -- while guaranteeing no (0, 0)
    (or any other degenerate placeholder) can ever reach a plot again.

    Used for VISUALIZATION LAYOUT ONLY. Every actual cost computation
    anywhere in this codebase uses `cost_matrix` directly, never these
    reconstructed coordinates -- MDS distances are an approximation of
    the true travel costs, not a substitute for them.

    Returns `existing_coords` unchanged if reconstruction fails, rather
    than None -- callers should never have to guard against a None
    coords array.
    """
    from sklearn.manifold import MDS

    with warnings.catch_warnings():
        # scikit-learn's own MDS emits several FutureWarnings about
        # upcoming default changes (n_init, init, dissimilarity ->
        # metric) that are irrelevant here -- we always pass
        # dissimilarity="precomputed" and n_init explicitly, and don't
        # want fixing the (0,0)-coordinate bug to trade it for a new
        # source of warning spam.
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
        try:
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                      normalized_stress=False, n_init=4)
        except TypeError:
            # Older scikit-learn versions don't accept normalized_stress.
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=4)

        try:
            return mds.fit_transform(cost_matrix)
        except Exception as e:
            warnings.warn(
                f"[{route_id}] MDS coordinate reconstruction failed ({e}); "
                f"falling back to the raw (possibly degenerate) coordinates. "
                f"Plots for this route may still be distorted."
            )
            return existing_coords


def compute_open_route_cost(tour, matrix):
    """
    Open TSP cost: sum of consecutive edge costs along `tour`, WITHOUT a
    return-to-start leg. This is the single source of truth for route
    cost -- run_amazon_experiment.py and compare_amazon_planned.py should
    both import this rather than defining their own local copies, so the
    formula can never silently drift between scripts again.
    """
    if not tour or len(tour) < 2:
        return 0.0
    return float(sum(matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))
