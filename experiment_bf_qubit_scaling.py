"""
experiment_bf_qubit_scaling.py

Standalone experiment: take ONE Amazon route, and run the FULL
receding-horizon heuristic algorithm (not an isolated subproblem -- the
actual end-to-end tour construction, algo_hybrid_bellmanford.py's
run_algo_hybrid_bf) at every window_size k from 2 up to k_max (default
20), to see what increasing the candidate-window size ("qubit count",
in QAOA terms) does to final tour quality.

--------------------------------------------------------------------------
WHY BELLMAN-FORD, NOT QAOA, FOR THIS EXPERIMENT
--------------------------------------------------------------------------
This is explicitly a THEORY experiment, not a claim about what QAOA
itself would achieve at k=20: real QAOA statevector simulation costs
2^(k^2), making k=20 (400 qubits) astronomically infeasible on any
classical simulator, and far beyond any current or near-term quantum
hardware too. Bellman-Ford/Held-Karp (cg_hybrid_bellmanford_sub.py's
solve_bellman_ford_subtour, reused here via
algo_hybrid_bellmanford.run_algo_hybrid_bf) is EXACT and costs only
O(2^k * k^2) -- still exponential, but with a base that makes k=20
tractable in minutes rather than never. Using it here answers a
DIFFERENT, well-posed question: "if the k-node sub-tour solver were
always exactly optimal (which QAOA is NOT, and doesn't get closer to by
adding qubits alone -- see algo_hybrid_LRWSQAOA.py's own FIX LOG,
'DEPTH CHANGE', for a directly relevant negative result), what would a
LARGER candidate window do to the final tour, holding the rest of the
receding-horizon construction fixed?" That's a genuinely useful
theoretical ceiling to know, even though no real QAOA circuit at k=20 is
attemptable -- it separates "what would more qubits even buy you if the
solver were perfect" from "does QAOA's own solve quality scale with
qubit count" (a question already answered, empirically, in the
negative, in algo_hybrid_LRWSQAOA.py's own revision history).

--------------------------------------------------------------------------
WHY batch_count = window_size FOR THIS EXPERIMENT SPECIFICALLY
--------------------------------------------------------------------------
Elsewhere in this codebase (run_experiment_ALL.py, algo_hybrid_
bellmanford.py's own typical usage), batch_count is usually much
smaller than the window size -- commit a few nodes, re-plan with a
fresh window, repeat. That means roughly (n-1)/batch_count solver calls
per route -- for batch_count=1 on a 150-stop route, ~150 calls. At
k=20, each Bellman-Ford call was measured (see cg_hybrid_bellmanford_
sub.py's own module docstring) at several tens of seconds -- 150 calls
at that cost would take HOURS for a single k value, and this experiment
sweeps 19 of them (k=2..20).

This experiment instead sets batch_count = window_size: commit the
ENTIRE resolved k-node window before re-planning. That cuts solver
calls to roughly ceil((n-1)/k) -- for k=20 on a 150-stop route, about
8 calls, not 150. This is a deliberate, documented choice specific to
keeping this scaling SWEEP tractable, not a claim that batch_count=
window_size is how you'd want to run this algorithm for actual route
quality (a smaller batch_count with more frequent re-planning is
generally better for that, exactly why it's the default elsewhere).

--------------------------------------------------------------------------
WHAT THIS EXPERIMENT DOES NOT COVER
--------------------------------------------------------------------------
Column generation (cg_hybrid_bellmanford_sub.py) is NOT swept here by
default. CG's pricing step calls the k-node solver O(n_iterations * n)
times per run (not O(n/k) like the heuristic here) -- even with
aggressively reduced n_iterations and max_pricing_nodes, sweeping CG
across k=2..20 would take substantially longer than the heuristic sweep
above. Pass --include-cg to attempt it anyway, with a small, explicit,
reduced iteration/pricing-node budget and a printed runtime warning
before it starts -- this is opt-in specifically because of that cost,
not because CG is unsupported.
"""

import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algo_hybrid_bellmanford import run_algo_hybrid_bf
from cg_hybrid_bellmanford_sub import run_cg_hybrid_bellmanford_sub


def run_qubit_scaling_experiment(
    data, k_max=20, batch_count_mode="equal_to_k", seed=2026,
    include_cg=False, cg_n_iterations=2, cg_max_pricing_nodes=8,
):
    """
    For ONE route (`data`, with an 'amazon_planned_tour' key for the
    baseline comparison), runs the full heuristic (run_algo_hybrid_bf)
    at every window_size k = 2 .. min(n-1, k_max), holding
    exploration_percent=0 (deterministic) and batch_count=window_size
    (see module docstring for why). Returns a list of per-k result
    dicts and prints a running table as it goes (this can take several
    minutes at the high end of k -- progress is shown, not silent).

    If include_cg=True, ALSO runs run_cg_hybrid_bellmanford_sub at each
    k with a reduced (cg_n_iterations, cg_max_pricing_nodes) budget --
    see module docstring for why this is opt-in.
    """
    from algo_data_loader import compute_open_route_cost

    matrix = data["matrix"]
    n = data["n_nodes"]
    route_id = data.get("route_id", "unknown")
    amazon_tour = data.get("amazon_planned_tour")
    amazon_cost = compute_open_route_cost(amazon_tour, matrix) if amazon_tour is not None else None

    k_max_effective = min(n - 1, k_max)
    k_values = list(range(2, k_max_effective + 1))

    print(f"=== Qubit-count (window_size) scaling experiment: route {route_id} ({n} stops) ===")
    if amazon_cost is not None:
        print(f"Amazon planned cost: {amazon_cost:,.2f}")
    if k_max_effective < k_max:
        print(f"NOTE: route only has {n-1} non-depot stops; testing k=2..{k_max_effective} "
              f"instead of the requested k_max={k_max}.")
    print(f"Testing k = 2..{k_max_effective}, batch_count=window_size (see module docstring for why), "
          f"exploration_percent=0 (deterministic).\n")

    results = []
    for k in k_values:
        t0 = time.time()
        res = run_algo_hybrid_bf(
            data, window_size=k, batch_count=k, exploration_percent=0.0,
            seed=seed, bf_max_k=k_max_effective,
        )
        elapsed = time.time() - t0
        cost = res["cost"]
        improvement_pct = (
            -100.0 * (cost - amazon_cost) / amazon_cost if amazon_cost else None
        )
        num_calls = -(-(n - 1) // k)  # ceil((n-1)/k)

        row = {
            "k": k, "cost": cost, "improvement_pct_vs_amazon": improvement_pct,
            "runtime_sec": elapsed, "num_solver_calls": num_calls, "algorithm": "heuristic_bf",
        }
        results.append(row)

        imp_str = f"{improvement_pct:+6.2f}%" if improvement_pct is not None else "   n/a "
        print(f"k={k:2d}: cost={cost:10.2f}  vs Amazon={imp_str}  "
              f"time={elapsed:8.3f}s  solver_calls~{num_calls}")

        if include_cg:
            t0 = time.time()
            try:
                cg_res = run_cg_hybrid_bellmanford_sub(
                    data, window_size=k, exploration_percent=0.0, seed=seed, bf_max_k=k_max_effective,
                    n_iterations=cg_n_iterations, max_pricing_nodes=min(cg_max_pricing_nodes, n),
                )
                cg_elapsed = time.time() - t0
                cg_cost = cg_res["cost"]
                cg_improvement_pct = (
                    -100.0 * (cg_cost - amazon_cost) / amazon_cost if amazon_cost else None
                )
                cg_row = {
                    "k": k, "cost": cg_cost, "improvement_pct_vs_amazon": cg_improvement_pct,
                    "runtime_sec": cg_elapsed, "num_solver_calls": cg_n_iterations * min(cg_max_pricing_nodes, n),
                    "algorithm": "cg_bf",
                }
                results.append(cg_row)
                cg_imp_str = f"{cg_improvement_pct:+6.2f}%" if cg_improvement_pct is not None else "   n/a "
                print(f"      [cg_bf, n_iterations={cg_n_iterations}, max_pricing_nodes="
                      f"{min(cg_max_pricing_nodes, n)}]: cost={cg_cost:10.2f}  "
                      f"vs Amazon={cg_imp_str}  time={cg_elapsed:8.3f}s")
            except Exception as e:
                warnings.warn(f"cg_bf failed at k={k}: {e}")

    return results


def plot_scaling_results(results, route_id, output_path):
    """Line chart(s): cost and improvement-vs-Amazon, both against k,
    one series per algorithm present in `results`. Saved as PNG."""
    heuristic_rows = sorted([r for r in results if r["algorithm"] == "heuristic_bf"], key=lambda r: r["k"])
    cg_rows = sorted([r for r in results if r["algorithm"] == "cg_bf"], key=lambda r: r["k"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor("#F7F8FA")

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", linewidth=0.6, color="#E4E6EA", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    h_k = [r["k"] for r in heuristic_rows]
    h_cost = [r["cost"] for r in heuristic_rows]
    ax1.plot(h_k, h_cost, "-o", color="#b5651d", label="Heuristic (Bellman-Ford)", markersize=4)
    ax2_data_present = heuristic_rows and heuristic_rows[0]["improvement_pct_vs_amazon"] is not None
    if ax2_data_present:
        h_imp = [r["improvement_pct_vs_amazon"] for r in heuristic_rows]
        ax2.plot(h_k, h_imp, "-o", color="#b5651d", label="Heuristic (Bellman-Ford)", markersize=4)

    if cg_rows:
        c_k = [r["k"] for r in cg_rows]
        c_cost = [r["cost"] for r in cg_rows]
        ax1.plot(c_k, c_cost, "-s", color="#8c1c13", label="CG (Bellman-Ford)", markersize=4)
        if ax2_data_present:
            c_imp = [r["improvement_pct_vs_amazon"] for r in cg_rows]
            ax2.plot(c_k, c_imp, "-s", color="#8c1c13", label="CG (Bellman-Ford)", markersize=4)

    ax1.set_xlabel("k (window_size / theoretical qubit count)")
    ax1.set_ylabel("Tour cost")
    ax1.set_title("Final tour cost vs. k")
    ax1.legend(fontsize=9)

    if ax2_data_present:
        ax2.axhline(0, color="#9A9EA6", linewidth=1, linestyle=":")
        ax2.set_xlabel("k (window_size / theoretical qubit count)")
        ax2.set_ylabel("Improvement vs. Amazon (%)")
        ax2.set_title("Improvement vs. Amazon planned, vs. k")
        ax2.legend(fontsize=9)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No Amazon baseline available for this route", ha="center", va="center",
                  transform=ax2.transAxes, color="#7A7E86")

    fig.suptitle(f"Theoretical effect of increasing k -- route {route_id}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Runs the full Bellman-Ford heuristic on one Amazon route at every window_size "
                    "k=2..k_max, to show the theoretical effect of increasing candidate-window size "
                    "('qubit count') on final tour quality -- see module docstring for why Bellman-Ford "
                    "(exact) stands in for 'what would happen if QAOA scaled perfectly', which QAOA "
                    "itself does not (see algo_hybrid_LRWSQAOA.py's own FIX LOG, 'DEPTH CHANGE')."
    )
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training")
    parser.add_argument("--route-id", type=str, default=None,
                        help="Specific route ID. If omitted, one is picked automatically (seeded), "
                             "or a synthetic route is used if the dataset isn't found.")
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--synthetic-n", type=int, default=60,
                        help="Size of the synthetic fallback route if no real dataset is found.")
    parser.add_argument("--output-dir", type=str, default="./bf_qubit_scaling_results")
    parser.add_argument("--include-cg", action="store_true",
                        help="ALSO sweep cg_bf at each k (reduced iteration/pricing-node budget -- "
                             "see module docstring for why this is opt-in and slower).")
    parser.add_argument("--cg-n-iterations", type=int, default=2)
    parser.add_argument("--cg-max-pricing-nodes", type=int, default=8)
    parser.add_argument("--yes", action="store_true", help="Skip the pre-flight confirmation prompt.")

    args = parser.parse_args()

    data = None
    if os.path.exists(args.data_dir) or os.path.exists("./data"):
        try:
            from algo_data_loader import AmazonDataLoader

            data_dir = args.data_dir if os.path.exists(args.data_dir) else "./data"
            loader = AmazonDataLoader(data_dir=data_dir)
            if loader.travel_times:
                all_route_ids = sorted(loader.travel_times.keys())
                route_id = args.route_id
                if route_id is None:
                    rng0 = np.random.default_rng(args.seed)
                    route_id = str(rng0.choice(all_route_ids))
                elif route_id not in all_route_ids:
                    raise ValueError(f"route_id '{route_id}' not found in '{data_dir}'.")

                extracted = loader.extract_single_route(route_id)
                matrix = np.array(extracted["matrix"])
                coords = np.array(extracted["coords"])
                if coords is None or np.all(coords == 0):
                    from sklearn.manifold import MDS
                    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=args.seed)
                    coords = mds.fit_transform(matrix)

                data = {
                    "route_id": extracted.get("route_id", route_id),
                    "n_nodes": extracted["n_nodes"],
                    "coords": coords,
                    "matrix": matrix,
                    "depot_idx": extracted.get("depot_idx", 0),
                    "amazon_planned_tour": extracted["amazon_planned_sequence"],
                }
                print(f"Loaded real route '{data['route_id']}' from '{data_dir}' "
                      f"({data['n_nodes']} stops).\n")
        except Exception as e:
            print(f"Could not load a real Amazon route ({e}); falling back to a synthetic route.\n")

    if data is None:
        rng0 = np.random.default_rng(args.seed)
        n = args.synthetic_n
        coords = rng0.uniform(0, 100, size=(n, 2))
        matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        synth_tour = [0]
        remaining = set(range(1, n))
        while remaining:
            last = synth_tour[-1]
            nxt = min(remaining, key=lambda x: matrix[last, x])
            synth_tour.append(nxt)
            remaining.remove(nxt)
        data = {
            "route_id": f"synthetic-{n}stops-seed{args.seed}",
            "n_nodes": n, "coords": coords, "matrix": matrix, "depot_idx": 0,
            "amazon_planned_tour": synth_tour,
        }
        print(f"No Amazon dataset found at '{args.data_dir}' -- using a synthetic "
              f"{n}-stop route instead (seed={args.seed}), with a nearest-neighbor tour "
              f"standing in for 'Amazon planned'.\n")

    n = data["n_nodes"]
    k_max_effective = min(n - 1, args.k_max)
    est_calls_at_kmax = -(-(n - 1) // k_max_effective)
    print(
        f"Pre-flight estimate: sweeping k=2..{k_max_effective}. The most expensive single k "
        f"(k={k_max_effective}) needs about {est_calls_at_kmax} exact Bellman-Ford solver call(s), "
        f"each of which can take on the order of tens of seconds at k~18-20 (see cg_hybrid_"
        f"bellmanford_sub.py's module docstring for measured numbers) -- the full sweep across all "
        f"k values is dominated by the last few k's cost, typically several minutes total.\n"
    )
    if not args.yes:
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            resp = "y"
        if resp != "y":
            print("Aborted.")
            raise SystemExit(0)

    results = run_qubit_scaling_experiment(
        data, k_max=args.k_max, seed=args.seed, include_cg=args.include_cg,
        cg_n_iterations=args.cg_n_iterations, cg_max_pricing_nodes=args.cg_max_pricing_nodes,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    import csv
    csv_path = os.path.join(args.output_dir, f"scaling_{data['route_id']}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "k", "cost", "improvement_pct_vs_amazon",
                                                "runtime_sec", "num_solver_calls"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults saved to: {csv_path}")

    plot_path = os.path.join(args.output_dir, f"scaling_{data['route_id']}.png")
    plot_scaling_results(results, data["route_id"], plot_path)
    print(f"Chart saved to: {plot_path}")
