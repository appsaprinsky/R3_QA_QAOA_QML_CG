"""
visualize_step_by_step_CG_REAL_DATA.py

Runs the exact same step-by-step CG diagnostic pipeline as
visualize_step_by_step_CG.py, but against ONE real Amazon Last-Mile
route instead of synthetic data. Does not reimplement any of the frame
drawing or algorithm-driving logic -- it imports and calls
visualize_cg_stepwise_execution() directly, so every fix made to that
function (opacity, exhaustive point/iteration coverage, dual-adjusted
QAOA matrix, etc.) applies here automatically with no risk of drift
between a synthetic-data script and a real-data script.

--------------------------------------------------------------------------
WHY THE DEFAULTS ARE DIFFERENT FROM visualize_step_by_step_CG.py
--------------------------------------------------------------------------
Real Amazon routes typically have 100-250 stops, not the 16-40 used in
the synthetic examples. visualize_cg_stepwise_execution()'s own default
is still "every point, every iteration, no sampling" (unchanged here,
that requirement is not being relaxed) -- but left at cg_hybrid_lrwsqaoa_sub.py's
global ITERATION_CG=10 default, "every point x every iteration" on a
200-stop route would mean up to 2000 pricing-detail frames, likely
hours of wall-clock time (QAOA statevector simulation scales with
2**(qubit_count**2) per subproblem, and every one of those 2000 frames
requires at least one real QAOA solve). That is not a hidden change of
behavior -- it is the honest cost of "every point, every iteration" at
real-route scale.

To make that cost visible and controllable rather than silently eaten,
this script:
  1. Defaults n_iterations to a smaller number (3) than the global
     ITERATION_CG=10, specifically for this script -- override with
     --n-iterations.
  2. Prints an explicit pre-flight estimate (route size x iterations x
     ~7 frame types) before starting, so the actual frame count is known
     up front, not discovered after waiting.
  3. Leaves max_detail_nodes and detail_iterations fully available as
     CLI flags (still None/unrestricted by default, matching the
     established requirement) so a real run can be scoped down
     deliberately -- e.g. --max-detail-nodes 20 -- without that being a
     silent default choice made on the user's behalf.
"""

import argparse
import os
import sys

import numpy as np

# --- CRITICAL CPU & THERMAL LIMITS (matches run_amazon_experiment.py / run_CG_experiment.py) ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

from algo_data_loader import AmazonDataLoader
from visualize_step_by_step_CG import visualize_cg_stepwise_execution, ITERATION_CG


def load_one_real_route(data_dir, route_id=None, seed=2026):
    """
    Loads exactly one real Amazon route in the same dict shape
    visualize_cg_stepwise_execution() (and run_cg_hybrid_lrwsqaoa_sub())
    expect: matrix, coords, depot_idx, n_nodes, route_id.

    If route_id is None, picks one route pseudo-randomly (seeded) from
    whatever routes are available in data_dir, rather than always the
    first -- so repeated runs without an explicit --route-id still
    sample different real routes across calls with different seeds,
    while a fixed seed reproduces the same pick.
    """
    if not os.path.exists(data_dir) and os.path.exists("./data"):
        data_dir = "./data"

    loader = AmazonDataLoader(data_dir=data_dir)
    if not loader.travel_times:
        raise FileNotFoundError(
            f"No route data found in '{data_dir}'. Ensure travel_times.json is available "
            f"(same data directory used by run_amazon_experiment.py / run_CG_experiment.py)."
        )

    all_route_ids = sorted(loader.travel_times.keys())
    if route_id is None:
        rng = np.random.default_rng(seed)
        route_id = str(rng.choice(all_route_ids))
    elif route_id not in all_route_ids:
        raise ValueError(
            f"route_id '{route_id}' not found in '{data_dir}'. "
            f"{len(all_route_ids)} routes available; pass --route-id with one of them, "
            f"or omit --route-id to pick one automatically."
        )

    extracted = loader.extract_single_route(route_id)
    matrix = np.array(extracted["matrix"])
    coords = np.array(extracted["coords"])

    if coords is None or np.all(coords == 0):
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed)
        coords = mds.fit_transform(matrix)

    data = {
        "route_id": extracted.get("route_id", route_id),
        "n_nodes": extracted["n_nodes"],
        "coords": coords,
        "matrix": matrix,
        "depot_idx": extracted.get("depot_idx", 0),
    }
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step CG diagnostic visualization on one real Amazon route."
    )
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training", help="Dataset directory")
    parser.add_argument("--route-id", type=str, default=None,
                         help="Specific route ID to use. If omitted, one is picked automatically (seeded).")
    parser.add_argument("--qubit-count", type=int, default=4)
    parser.add_argument("--exploration-percent", type=float, default=0.1)
    parser.add_argument("--xy-mixer", action="store_true")
    parser.add_argument("--no-only-improving-columns", action="store_true",
                         help="Disable the reduced-cost filter (keep every priced truncation, not just improving ones).")
    parser.add_argument("--n-iterations", type=int, default=3,
                         help="Pricing iterations for THIS script (default 3, deliberately lower than the "
                              f"global ITERATION_CG={ITERATION_CG} default -- see module docstring for why).")
    parser.add_argument("--max-detail-nodes", type=int, default=None,
                         help="Cap detail frames to this many points per rendered iteration. "
                              "Default None = every point (matches the established requirement) -- "
                              "set this explicitly to scope down a large real route.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default="./cg_visualizations_real_data")
    parser.add_argument("--yes", action="store_true", help="Skip the pre-flight confirmation prompt.")

    args = parser.parse_args()

    print(f"Loading a real Amazon route from '{args.data_dir}'"
          f"{f' (route_id={args.route_id})' if args.route_id else ' (auto-selected)'}...")
    data = load_one_real_route(args.data_dir, route_id=args.route_id, seed=args.seed)
    n = data["n_nodes"]
    print(f"  route_id={data['route_id']}, {n} stops (including depot)")

    detail_node_count = n if args.max_detail_nodes is None else min(args.max_detail_nodes, n)
    est_pricing_frames = detail_node_count * args.n_iterations
    print(
        f"\nPre-flight estimate:\n"
        f"  {detail_node_count} points x {args.n_iterations} iteration(s) = "
        f"{est_pricing_frames} pricing-detail frames (plus ~7 summary frames).\n"
        f"  Each point-iteration involves at least one real QAOA statevector solve --\n"
        f"  this can be slow at real-route scale. Use --max-detail-nodes and/or\n"
        f"  --n-iterations to scope this down if needed.\n"
    )
    if not args.yes:
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            resp = "y"  # non-interactive context (e.g. piped input) -- proceed rather than hang
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    result = visualize_cg_stepwise_execution(
        data,
        qubit_count=args.qubit_count,
        exploration_percent=args.exploration_percent,
        xy_mixer=args.xy_mixer,
        only_improving_columns=not args.no_only_improving_columns,
        n_iterations=args.n_iterations,
        max_detail_nodes=args.max_detail_nodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"\nDone. Final tour cost: {result['final_cost']:.2f} "
          f"(raw pre-2opt: {result['raw_cost']:.2f}). "
          f"Frames saved under '{args.output_dir}/'")


if __name__ == "__main__":
    main()
