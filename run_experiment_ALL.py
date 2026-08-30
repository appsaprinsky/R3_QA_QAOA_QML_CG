"""
run_experiment_ALL.py

Unified experimental runner. For the SAME route sample (same
`--data-dir`/`--num-routes`/`--seed`), this script now runs FOUR
algorithms, each treated as fully independent:

  - heuristic_qaoa : WS-LR-QAOA receding-horizon heuristic
                      (run_algo_hybrid_2_5, algo_hybrid_LRWSQAOA.py)
  - cg_qaoa        : Column Generation, Truncated Dual-Embedded QAOA
                      Pricing (run_cg_hybrid_lrwsqaoa_sub,
                      cg_hybrid_lrwsqaoa_sub.py)
  - heuristic_bf   : the SAME receding-horizon construction as
                      heuristic_qaoa, but the k-node sub-tour is solved
                      EXACTLY via Bellman-Ford/Held-Karp instead of QAOA
                      (run_algo_hybrid_bf, algo_hybrid_bellmanford.py)
  - cg_bf          : the SAME column-generation master problem as
                      cg_qaoa, but pricing is solved EXACTLY via
                      Bellman-Ford/Held-Karp instead of QAOA
                      (run_cg_hybrid_bellmanford_sub,
                      cg_hybrid_bellmanford_sub.py)

against the same Amazon Planned baseline, writing ONE combined set of
comparison artifacts:

    <output-dir>/
      experiment_ALL_detailed_results.csv        one row per
        (route, algorithm, parameter combination) -- long format, an
        "algorithm" column (one of the four names above) distinguishes
        rows. A single "qubit_count" column is used for ALL FOUR
        algorithms' candidate-window size k -- see "K PINNING" below for
        why -- so filtering/comparing across algorithms on k is a single
        column, not four different ones. Columns that don't apply to a
        given algorithm (batch_count for cg_* rows, xy_mixer /
        only_improving_columns for heuristic_* rows, bf_max_k for *_qaoa
        rows) are left blank on that row.
      experiment_ALL_summary_by_parameters.csv   grouped by algorithm +
        that algorithm's own parameter columns: mean cost, mean
        improvement % vs Amazon, win rate %, mean runtime, plus the
        cg-only diagnostics (segments selected, iterations run) where
        algorithm is cg_qaoa or cg_bf.
      experiment_ALL_route_comparison.csv        ONE row per route:
        Amazon cost side by side with the BEST result from EACH of the
        four algorithms' own grids, their runtimes, their param
        strings, each one's improvement % over Amazon, AND an overall
        "winner" among {Amazon, heuristic_qaoa, cg_qaoa, heuristic_bf,
        cg_bf} -- the single file to open for "which approach actually
        won, per route."
      visualise_experiments_ALL/
        heuristic_qaoa_vs_amazon/{plots_with_depot,plots_without_depot}/
        cg_qaoa_vs_amazon/{plots_with_depot,plots_without_depot}/
        heuristic_bf_vs_amazon/{plots_with_depot,plots_without_depot}/
        cg_bf_vs_amazon/{plots_with_depot,plots_without_depot}/
        (PNG by default -- see --plot-formats -- via
        plot_publication.generate_overall_visualizations, the SAME
        2-panel figure function every other experiment script uses,
        called once per algorithm per route.)

--------------------------------------------------------------------------
K PINNING (explicit requirement: "make sure K is equal to number of
qubit there")
--------------------------------------------------------------------------
heuristic_bf's candidate-window sizes are NOT configured separately --
they are exactly `heuristic_qubit_counts` (the same list used for
heuristic_qaoa). Likewise cg_bf's window sizes are exactly
`cg_qubit_counts`. There is no --bf-window-sizes flag; this is
deliberate, so a comparison run is always apples-to-apples at the same k
across the QAOA and Bellman-Ford version of each algorithm family --
not an independently-tunable, possibly-mismatched parameter. The one
BF-specific knob is `bf_max_k` (default 18, see cg_hybrid_bellmanford_
sub.py's module docstring for the measured runtimes behind that
default) -- a SAFETY CAP, not a search parameter: if a configured qubit
count exceeds it, that combination is recorded as an error for the BF
algorithm (not silently skipped, not a crash) rather than attempting a
run that could take hours. QAOA has no equivalent cap here because its
own qubit-count-driven statevector cost already makes large k
impractical long before this would matter.

Design notes carried over from the two-algorithm version
----------------------------------------------------------
* None of the four run_* functions, the cost metric, or the route-
  sampling function are reimplemented here -- all imported from their
  single source of truth.
* Each algorithm is swept over its OWN full grid, independently, for
  every route (they don't share a parameter space -- heuristics have
  batch_count, CG has only_improving_columns, only QAOA variants have
  xy_mixer, only BF variants have bf_max_k) -- combined only at the
  reporting layer, not by forcing a shared parameter space that doesn't
  exist.
* Visualization defaults to one figure per DISTINCT qubit_count tested,
  per algorithm, per route (`--plot-granularity per_qubit`), not one
  per grid combination -- see the two-algorithm version's original FIX
  LOG (preserved in git history / prior revisions of this file) for why.
"""

# --- CRITICAL CPU & THERMAL LIMITS ---
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import gc
import itertools
import time
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from algo_data_loader import AmazonDataLoader, compute_open_route_cost
from algo_hybrid_LRWSQAOA import run_algo_hybrid_2_5
from algo_hybrid_bellmanford import run_algo_hybrid_bf
from cg_hybrid_lrwsqaoa_sub import run_cg_hybrid_lrwsqaoa_sub, ITERATION_CG
from cg_hybrid_bellmanford_sub import run_cg_hybrid_bellmanford_sub
from plot_publication import generate_overall_visualizations


# ---------------------------------------------------------------------
# Route sampling -- single source of truth for all four algorithms.
# ---------------------------------------------------------------------

def get_amazon_dataset_sample(data_dir="./almrrc2021-data-training", num_routes=10, seed=2026):
    """Loads real ALMRRC routes. See algo_data_loader.py's own FIX LOG:
    the `np.all(coords == 0)` check below is a redundant safety net --
    the primary coordinate-repair fix lives in extract_single_route()
    itself now."""
    if not os.path.exists(data_dir) and os.path.exists("./data"):
        data_dir = "./data"

    loader = AmazonDataLoader(data_dir=data_dir)
    if not loader.travel_times:
        raise FileNotFoundError(
            f"No route data found in '{data_dir}'. Ensure travel_times.json is available."
        )

    all_route_ids = sorted(list(loader.travel_times.keys()))
    if num_routes is not None and num_routes < len(all_route_ids):
        rng = np.random.default_rng(seed)
        selected_route_ids = rng.choice(all_route_ids, size=num_routes, replace=False).tolist()
    else:
        selected_route_ids = all_route_ids

    print(f"--> Selected {len(selected_route_ids)} route(s) out of {len(all_route_ids)} total routes.")

    dataset = []
    for rid in selected_route_ids:
        extracted = loader.extract_single_route(rid)
        matrix = np.array(extracted["matrix"])
        coords = np.array(extracted["coords"])
        if coords is None or np.all(coords == 0):
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed)
            coords = mds.fit_transform(matrix)
        dataset.append({
            "route_id": extracted.get("route_id", rid),
            "n_nodes": extracted["n_nodes"],
            "coords": coords,
            "matrix": matrix,
            "depot_idx": extracted.get("depot_idx", 0),
            "amazon_planned_tour": extracted["amazon_planned_sequence"],
        })
    return dataset


# ---------------------------------------------------------------------
# Per-route, per-algorithm grid execution
# ---------------------------------------------------------------------

def _best_per_qubit_count(records):
    """{qubit_count: best_record} -- cheapest algo_cost per distinct k,
    marginalizing over every other parameter. Records with an error are
    skipped. Shared across all four algorithms since they all populate
    the same "qubit_count" column (see module docstring, K PINNING)."""
    best = {}
    for rec in records:
        if rec.get("error"):
            continue
        q = rec["qubit_count"]
        if q not in best or rec["algo_cost"] < best[q]["algo_cost"]:
            best[q] = rec
    return best


def _base_record(data, algorithm, q, exp, param_str):
    return {
        "route_id": data["route_id"], "n_nodes": data["n_nodes"], "algorithm": algorithm,
        "qubit_count": q, "exploration_percent": exp, "batch_count": None, "xy_mixer": None,
        "only_improving_columns": None, "bf_max_k": None,
        "amazon_cost": None, "algo_cost": None, "cost_diff_abs": None, "cost_diff_pct": None,
        "improvement_pct": None, "runtime_sec": None,
        "num_iterations_run": None, "num_pool_columns_final": None, "num_segments_selected": None,
        "final_master_status": None, "pre_2opt_cost": None, "param_str": param_str, "error": None,
    }


def _finalize_record(rec, amazon_cost, cost, elapsed, tour, keep_all_tours):
    cost_diff_abs = cost - amazon_cost
    cost_diff_pct = (cost_diff_abs / amazon_cost) * 100.0 if amazon_cost else 0.0
    rec.update({
        "amazon_cost": round(amazon_cost, 2), "algo_cost": round(cost, 2),
        "cost_diff_abs": round(cost_diff_abs, 2), "cost_diff_pct": round(cost_diff_pct, 2),
        "improvement_pct": round(-cost_diff_pct, 2), "runtime_sec": round(elapsed, 3),
    })
    if keep_all_tours:
        rec["_tour"] = tour
    return rec


def _run_heuristic_qaoa_grid_for_route(data, amazon_cost, param_grid, seed, keep_all_tours=False):
    """param_grid: (qubit_count, exploration_percent, batch_count, xy_mixer)"""
    matrix = data["matrix"]
    records, best = [], None
    for q, exp, b, xy in param_grid:
        param_str = f"h_q{q}_exp{int(exp*100)}_b{b}_xy{1 if xy else 0}"
        t0 = time.time()
        res = run_algo_hybrid_2_5(data, qubit_count=q, exploration_percent=exp,
                                   batch_count=b, xy_mixer=xy, seed=seed)
        elapsed = time.time() - t0
        tour = res["tour"]
        cost = compute_open_route_cost(tour, matrix)
        rec = _base_record(data, "heuristic_qaoa", q, exp, param_str)
        rec["batch_count"], rec["xy_mixer"] = b, xy
        rec = _finalize_record(rec, amazon_cost, cost, elapsed, tour, keep_all_tours)
        records.append(rec)
        if best is None or cost < best["algo_cost"]:
            best = dict(rec)
            best["_tour"] = tour
    return records, best


def _run_heuristic_bf_grid_for_route(data, amazon_cost, param_grid, seed, bf_max_k, keep_all_tours=False):
    """param_grid: (window_size, exploration_percent, batch_count) -- no
    xy_mixer (nothing quantum here). window_size populates the shared
    "qubit_count" column (see module docstring, K PINNING).

    NOTE: run_algo_hybrid_bf() itself SILENTLY CLAMPS window_size down
    to bf_max_k rather than raising when window_size > bf_max_k (that's
    reasonable default behavior for that file used standalone, where
    "run at whatever's feasible" is a sensible fallback) -- but here,
    where the whole point is an apples-to-apples comparison at an
    EXACT, pinned k against heuristic_qaoa, silently running at a
    smaller k than requested (while still being labeled with the
    originally-requested qubit_count in every CSV row) would silently
    break that guarantee. So this function checks w > bf_max_k itself,
    BEFORE calling run_algo_hybrid_bf, and records it as an explicit
    error instead of letting the silent clamp happen unnoticed.
    """
    matrix = data["matrix"]
    records, best = [], None
    for w, exp, b in param_grid:
        param_str = f"hbf_w{w}_exp{int(exp*100)}_b{b}"
        if w > bf_max_k:
            rec = _base_record(data, "heuristic_bf", w, exp, param_str)
            rec["batch_count"], rec["bf_max_k"] = b, bf_max_k
            rec["runtime_sec"] = 0.0
            rec["error"] = (
                f"window_size={w} exceeds bf_max_k={bf_max_k}; skipped rather than silently "
                f"clamped down (which would break the k-pinned comparison against heuristic_qaoa "
                f"at the same nominal qubit_count)."
            )
            records.append(rec)
            continue
        t0 = time.time()
        res = run_algo_hybrid_bf(data, window_size=w, exploration_percent=exp,
                                  batch_count=b, seed=seed, bf_max_k=bf_max_k)
        elapsed = time.time() - t0
        tour = res["tour"]
        cost = compute_open_route_cost(tour, matrix)
        rec = _base_record(data, "heuristic_bf", w, exp, param_str)
        rec["batch_count"], rec["bf_max_k"] = b, bf_max_k
        rec = _finalize_record(rec, amazon_cost, cost, elapsed, tour, keep_all_tours)
        records.append(rec)
        if best is None or cost < best["algo_cost"]:
            best = dict(rec)
            best["_tour"] = tour
    return records, best


def _run_cg_qaoa_grid_for_route(data, amazon_cost, param_grid, max_pricing_nodes, n_iterations,
                                 time_limit, seed, keep_all_tours=False):
    """param_grid: (qubit_count, exploration_percent, xy_mixer, only_improving_columns)"""
    matrix = data["matrix"]
    records, best = [], None
    for q, exp, xy, only_improving in param_grid:
        param_str = f"cg_q{q}_exp{int(exp*100)}_xy{1 if xy else 0}_imp{1 if only_improving else 0}"
        t0 = time.time()
        try:
            res = run_cg_hybrid_lrwsqaoa_sub(
                data, qubit_count=q, exploration_percent=exp, xy_mixer=xy,
                only_improving_columns=only_improving, max_pricing_nodes=max_pricing_nodes,
                n_iterations=n_iterations, time_limit=time_limit, seed=seed,
            )
        except Exception as e:
            elapsed = time.time() - t0
            rec = _base_record(data, "cg_qaoa", q, exp, param_str)
            rec["xy_mixer"], rec["only_improving_columns"] = xy, only_improving
            rec["runtime_sec"], rec["error"] = round(elapsed, 3), str(e)
            records.append(rec)
            continue
        elapsed = time.time() - t0
        tour = res["tour"]
        cost = compute_open_route_cost(tour, matrix)
        diag = res["cg_diagnostics"]
        rec = _base_record(data, "cg_qaoa", q, exp, param_str)
        rec["xy_mixer"], rec["only_improving_columns"] = xy, only_improving
        rec.update({
            "num_iterations_run": diag["num_iterations_run"],
            "num_pool_columns_final": diag["num_pool_columns_final"],
            "num_segments_selected": diag["num_segments_selected"],
            "final_master_status": diag["final_master_status"],
            "pre_2opt_cost": round(diag["pre_2opt_cost"], 2),
        })
        rec = _finalize_record(rec, amazon_cost, cost, elapsed, tour, keep_all_tours)
        records.append(rec)
        if best is None or cost < best["algo_cost"]:
            best = dict(rec)
            best["_tour"] = tour
    return records, best


def _run_cg_bf_grid_for_route(data, amazon_cost, param_grid, max_pricing_nodes, n_iterations,
                               time_limit, seed, bf_max_k, keep_all_tours=False):
    """param_grid: (window_size, exploration_percent, only_improving_columns) --
    no xy_mixer. window_size populates the shared "qubit_count" column.
    See _run_heuristic_bf_grid_for_route's docstring for why w > bf_max_k
    is checked explicitly here rather than relying on the underlying
    function's own silent clamp."""
    matrix = data["matrix"]
    records, best = [], None
    for w, exp, only_improving in param_grid:
        param_str = f"cgbf_w{w}_exp{int(exp*100)}_imp{1 if only_improving else 0}"
        if w > bf_max_k:
            rec = _base_record(data, "cg_bf", w, exp, param_str)
            rec["only_improving_columns"], rec["bf_max_k"] = only_improving, bf_max_k
            rec["runtime_sec"] = 0.0
            rec["error"] = (
                f"window_size={w} exceeds bf_max_k={bf_max_k}; skipped rather than silently "
                f"clamped down (which would break the k-pinned comparison against cg_qaoa at the "
                f"same nominal qubit_count)."
            )
            records.append(rec)
            continue
        t0 = time.time()
        try:
            res = run_cg_hybrid_bellmanford_sub(
                data, window_size=w, exploration_percent=exp, only_improving_columns=only_improving,
                max_pricing_nodes=max_pricing_nodes, n_iterations=n_iterations, time_limit=time_limit,
                seed=seed, bf_max_k=bf_max_k,
            )
        except Exception as e:
            elapsed = time.time() - t0
            rec = _base_record(data, "cg_bf", w, exp, param_str)
            rec["only_improving_columns"], rec["bf_max_k"] = only_improving, bf_max_k
            rec["runtime_sec"], rec["error"] = round(elapsed, 3), str(e)
            records.append(rec)
            continue
        elapsed = time.time() - t0
        tour = res["tour"]
        cost = compute_open_route_cost(tour, matrix)
        diag = res["cg_diagnostics"]
        rec = _base_record(data, "cg_bf", w, exp, param_str)
        rec["only_improving_columns"], rec["bf_max_k"] = only_improving, bf_max_k
        rec.update({
            "num_iterations_run": diag["num_iterations_run"],
            "num_pool_columns_final": diag["num_pool_columns_final"],
            "num_segments_selected": diag["num_segments_selected"],
            "final_master_status": diag["final_master_status"],
            "pre_2opt_cost": round(diag["pre_2opt_cost"], 2),
        })
        rec = _finalize_record(rec, amazon_cost, cost, elapsed, tour, keep_all_tours)
        records.append(rec)
        if best is None or cost < best["algo_cost"]:
            best = dict(rec)
            best["_tour"] = tour
    return records, best


# ---------------------------------------------------------------------
# Main grid driver
# ---------------------------------------------------------------------

# (algorithm key, display label, plot color, viz subfolder name)
_ALGO_DISPLAY = {
    "heuristic_qaoa": ("Heuristic (WS-LR-QAOA)", "#1a6b1a", "heuristic_qaoa_vs_amazon"),
    "cg_qaoa":        ("CG (TDE-QP)",             "#6a3d9a", "cg_qaoa_vs_amazon"),
    "heuristic_bf":   ("Heuristic (Bellman-Ford)", "#b5651d", "heuristic_bf_vs_amazon"),
    "cg_bf":          ("CG (Bellman-Ford)",        "#8c1c13", "cg_bf_vs_amazon"),
}


def run_all_experiment_grid(
    data_dir="./almrrc2021-data-training",
    num_routes=10,
    output_dir="./experiment_results_all",
    seed=2026,
    # --- heuristic_qaoa / heuristic_bf grid (k PINNED across both -- see module docstring) ---
    heuristic_qubit_counts=(2, 3),
    heuristic_exploration_percents=(0.0, 0.2),
    heuristic_batch_counts=(1, 2, 3, 4),
    heuristic_xy_mixers=(False,),
    # --- cg_qaoa / cg_bf grid (k PINNED across both) ---
    cg_qubit_counts=(2, 3),
    cg_exploration_percents=(0.0, 0.2),
    cg_xy_mixers=(False,),
    cg_only_improving_columns_options=(True,),
    cg_max_pricing_nodes=None,
    cg_n_iterations=ITERATION_CG,
    cg_time_limit=60,
    # --- Bellman-Ford-only ---
    bf_max_k=18,
    run_bf_algorithms=True,
    # --- Visualization ---
    generate_plots=True,
    plot_granularity="per_qubit",   # "best" | "per_qubit" | "all"
    plot_formats=("png",),
    plot_max_routes=None,
):
    """Runs all four algorithms' full grids, for every route in the
    sample, and writes the combined CSVs + comparison figures described
    in the module docstring. Set run_bf_algorithms=False to run only
    the original two (heuristic_qaoa, cg_qaoa), e.g. for a faster run."""
    if plot_granularity not in ("best", "per_qubit", "all"):
        raise ValueError(f"plot_granularity must be 'best', 'per_qubit', or 'all', got {plot_granularity!r}")

    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualise_experiments_ALL")
    viz_subdirs = {key: os.path.join(viz_dir, sub) for key, (_, _, sub) in _ALGO_DISPLAY.items()}

    routes_data = get_amazon_dataset_sample(data_dir=data_dir, num_routes=num_routes, seed=seed)

    h_qaoa_grid = [
        (q, exp, b, xy) for q, exp, b, xy in itertools.product(
            heuristic_qubit_counts, heuristic_exploration_percents,
            heuristic_batch_counts, heuristic_xy_mixers,
        ) if b <= q
    ]
    cg_qaoa_grid = list(itertools.product(
        cg_qubit_counts, cg_exploration_percents, cg_xy_mixers, cg_only_improving_columns_options,
    ))
    # K PINNED: reuse heuristic_qubit_counts / cg_qubit_counts directly
    # as the BF window-size grids -- no separate BF k parameter exists.
    h_bf_grid = [
        (w, exp, b) for w, exp, b in itertools.product(
            heuristic_qubit_counts, heuristic_exploration_percents, heuristic_batch_counts,
        ) if b <= w
    ]
    cg_bf_grid = list(itertools.product(
        cg_qubit_counts, cg_exploration_percents, cg_only_improving_columns_options,
    ))
    if not run_bf_algorithms:
        h_bf_grid, cg_bf_grid = [], []

    total_combos = len(h_qaoa_grid) + len(cg_qaoa_grid) + len(h_bf_grid) + len(cg_bf_grid)
    print(
        f"\n=== Starting UNIFIED Grid Search Benchmark (4 algorithms) ==="
        f"\n  Routes Sampled           : {len(routes_data)}"
        f"\n  heuristic_qaoa Combos    : {len(h_qaoa_grid)}"
        f"\n  cg_qaoa Combos           : {len(cg_qaoa_grid)}"
        f"\n  heuristic_bf Combos      : {len(h_bf_grid)}  (k pinned to heuristic_qubit_counts)"
        f"\n  cg_bf Combos             : {len(cg_bf_grid)}  (k pinned to cg_qubit_counts)"
        f"\n  bf_max_k (safety cap)    : {bf_max_k}"
        f"\n  Plot Granularity         : {plot_granularity}  (formats: {', '.join(plot_formats)})"
        f"\n  Total Algorithm Runs     : {len(routes_data) * total_combos}\n"
    )

    all_records = []
    comparison_rows = []
    t_start_grid = time.time()

    for r_idx, data in enumerate(routes_data, 1):
        route_id = data["route_id"]
        matrix = data["matrix"]
        amazon_tour = data["amazon_planned_tour"]
        amazon_cost = compute_open_route_cost(amazon_tour, matrix)

        t_route_start = time.time()
        will_plot_this_route = generate_plots and (plot_max_routes is None or r_idx <= plot_max_routes)
        keep_all = will_plot_this_route and plot_granularity in ("per_qubit", "all")

        hq_records, hq_best = _run_heuristic_qaoa_grid_for_route(
            data, amazon_cost, h_qaoa_grid, seed, keep_all_tours=keep_all)
        cq_records, cq_best = _run_cg_qaoa_grid_for_route(
            data, amazon_cost, cg_qaoa_grid, cg_max_pricing_nodes, cg_n_iterations, cg_time_limit, seed,
            keep_all_tours=keep_all)
        hb_records, hb_best = _run_heuristic_bf_grid_for_route(
            data, amazon_cost, h_bf_grid, seed, bf_max_k, keep_all_tours=keep_all)
        cb_records, cb_best = _run_cg_bf_grid_for_route(
            data, amazon_cost, cg_bf_grid, cg_max_pricing_nodes, cg_n_iterations, cg_time_limit, seed,
            bf_max_k, keep_all_tours=keep_all)

        by_algo = {
            "heuristic_qaoa": (hq_records, hq_best), "cg_qaoa": (cq_records, cq_best),
            "heuristic_bf": (hb_records, hb_best), "cg_bf": (cb_records, cb_best),
        }
        for records, _ in by_algo.values():
            all_records.extend(records)

        def _fmt(best):
            return f"{best['algo_cost']:.2f} ({best['param_str']})" if best else "N/A"

        print(
            f"[{r_idx}/{len(routes_data)}] Route {route_id} | Amazon: {amazon_cost:.2f} | "
            f"h_qaoa: {_fmt(hq_best)} | cg_qaoa: {_fmt(cq_best)} | "
            f"h_bf: {_fmt(hb_best)} | cg_bf: {_fmt(cb_best)} | "
            f"Route time: {time.time() - t_route_start:.2f}s"
        )

        row = {"route_id": route_id, "n_nodes": data["n_nodes"], "amazon_cost": round(amazon_cost, 2)}
        best_costs = {"Amazon": amazon_cost}
        for algo_key, (_, best) in by_algo.items():
            row[f"best_{algo_key}_cost"] = best["algo_cost"] if best else None
            row[f"best_{algo_key}_params"] = best["param_str"] if best else None
            row[f"best_{algo_key}_runtime_sec"] = best["runtime_sec"] if best else None
            row[f"{algo_key}_improvement_pct_vs_amazon"] = best["improvement_pct"] if best else None
            if best:
                best_costs[algo_key] = best["algo_cost"]
        row["winner"] = min(best_costs, key=best_costs.get)
        comparison_rows.append(row)

        if will_plot_this_route:
            for algo_key, (records, best) in by_algo.items():
                algo_label, algo_color, _ = _ALGO_DISPLAY[algo_key]
                out_dir = viz_subdirs[algo_key]
                if plot_granularity == "all":
                    for rec in records:
                        if rec.get("error"):
                            continue
                        generate_overall_visualizations(
                            data, rec["_tour"], rec["algo_cost"], rec["param_str"], out_dir,
                            algo_label=algo_label, algo_color=algo_color, formats=plot_formats)
                elif plot_granularity == "per_qubit":
                    for q, rec in sorted(_best_per_qubit_count(records).items()):
                        generate_overall_visualizations(
                            data, rec["_tour"], rec["algo_cost"], rec["param_str"], out_dir,
                            algo_label=algo_label, algo_color=algo_color, formats=plot_formats)
                else:  # "best"
                    if best:
                        generate_overall_visualizations(
                            data, best["_tour"], best["algo_cost"], best["param_str"], out_dir,
                            algo_label=algo_label, algo_color=algo_color, formats=plot_formats)

        gc.collect()

    # ------------------ Write combined CSVs ------------------
    df_all = pd.DataFrame(all_records).drop(columns=["_tour"], errors="ignore")
    detailed_csv = os.path.join(output_dir, "experiment_ALL_detailed_results.csv")
    df_all.to_csv(detailed_csv, index=False)

    summary_frames = []
    group_cols_by_algo = {
        "heuristic_qaoa": ["algorithm", "qubit_count", "exploration_percent", "batch_count", "xy_mixer"],
        "heuristic_bf":   ["algorithm", "qubit_count", "exploration_percent", "batch_count"],
        "cg_qaoa":        ["algorithm", "qubit_count", "exploration_percent", "xy_mixer", "only_improving_columns"],
        "cg_bf":          ["algorithm", "qubit_count", "exploration_percent", "only_improving_columns"],
    }
    for algo_key, group_cols in group_cols_by_algo.items():
        df_sub = df_all[(df_all["algorithm"] == algo_key) & df_all["error"].isna()]
        if len(df_sub) == 0:
            continue
        agg_kwargs = dict(
            mean_amazon_cost=("amazon_cost", "mean"),
            mean_algo_cost=("algo_cost", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            win_rate_pct=("improvement_pct", lambda x: (x > 0).mean() * 100),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        if algo_key in ("cg_qaoa", "cg_bf"):
            agg_kwargs["mean_segments_selected"] = ("num_segments_selected", "mean")
            agg_kwargs["mean_iterations_run"] = ("num_iterations_run", "mean")
        summary_frames.append(df_sub.groupby(group_cols).agg(**agg_kwargs).reset_index())
    df_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_csv = os.path.join(output_dir, "experiment_ALL_summary_by_parameters.csv")
    df_summary.to_csv(summary_csv, index=False)

    df_comparison = pd.DataFrame(comparison_rows)
    comparison_csv = os.path.join(output_dir, "experiment_ALL_route_comparison.csv")
    df_comparison.to_csv(comparison_csv, index=False)

    print("\n" + "=" * 100)
    print("                              UNIFIED BENCHMARK SUMMARY (by parameters)                          ")
    print("=" * 100)
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
    print("=" * 100)
    print("\n" + "=" * 100)
    print("                        PER-ROUTE COMPARISON (best config per algorithm)                        ")
    print("=" * 100)
    print(df_comparison.to_string(index=False))
    print("=" * 100)
    if not df_comparison.empty:
        win_counts = df_comparison["winner"].value_counts()
        print(f"\nWinner counts across {len(df_comparison)} routes:\n{win_counts.to_string()}")
    print(f"\n--> Detailed results   : {detailed_csv}")
    print(f"--> Parameter summary  : {summary_csv}")
    print(f"--> Route comparison   : {comparison_csv}")
    print(f"--> Visualizations     : {viz_dir}")
    print(f"--> Total Execution Time: {time.time() - t_start_grid:.2f}s\n")

    return df_all, df_summary, df_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all four algorithms (heuristic_qaoa, cg_qaoa, heuristic_bf, cg_bf) against "
                    "the Amazon baseline, on the same route sample, in one pass. heuristic_bf / cg_bf "
                    "use the SAME k values as heuristic_qaoa / cg_qaoa respectively -- see module "
                    "docstring, 'K PINNING'."
    )
    parser.add_argument("--data-dir", type=str, default="./almrrc2021-data-training", help="Dataset directory")
    parser.add_argument("--num-routes", type=int, default=2, help="Number of routes to sample")
    parser.add_argument("--output-dir", type=str, default="./experiment_results_all", help="Output directory for CSVs & plots")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (shared route sample + algorithm seeding)")
    parser.add_argument("--no-plots", action="store_true", help="Disable generating visual plots entirely")
    parser.add_argument("--no-bf", action="store_true",
                        help="Skip heuristic_bf and cg_bf entirely -- run only the original two "
                             "QAOA algorithms (faster).")
    parser.add_argument("--bf-max-k", type=int, default=18,
                        help="Safety cap on k for the Bellman-Ford algorithms (default 18 -- see "
                             "cg_hybrid_bellmanford_sub.py's module docstring for measured runtimes). "
                             "A configured qubit count above this is recorded as an error for the BF "
                             "algorithms, not silently skipped or crashed on.")
    parser.add_argument("--plot-granularity", type=str, default="per_qubit", choices=["best", "per_qubit", "all"],
                        help="'best': one overall-best figure per algorithm per route. 'per_qubit' (default): "
                             "one figure per DISTINCT qubit count tested per algorithm per route. 'all': every "
                             "single grid combination (MANY files above a handful of routes).")
    parser.add_argument("--plot-all-combos", action="store_true",
                        help="Deprecated alias for --plot-granularity all (kept for backward compatibility).")
    parser.add_argument("--plot-formats", type=str, nargs="+", default=["png"], choices=["png", "pdf"],
                        help="Which format(s) to save each figure in (default: png only).")
    parser.add_argument("--plot-max-routes", type=int, default=None,
                        help="Only generate plots for the first N routes in the sample (all routes still get "
                             "full CSV results).")

    # --- heuristic_qaoa / heuristic_bf grid (k pinned across both) ---
    parser.add_argument("--h-qubit-counts", type=int, nargs="+", default=[2, 3],
                        help="Qubit budgets / Bellman-Ford window sizes to sweep for BOTH "
                             "heuristic_qaoa and heuristic_bf (same values, see K PINNING).")
    parser.add_argument("--h-exploration-percents", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--h-batch-counts", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Filtered to b <= k per combo, for both heuristic_qaoa and heuristic_bf.")
    parser.add_argument("--h-xy-mixer", action="store_true",
                        help="Also sweep xy_mixer=True for heuristic_qaoa only (no effect on heuristic_bf, "
                             "which has no mixer).")

    # --- cg_qaoa / cg_bf grid (k pinned across both) ---
    parser.add_argument("--cg-qubit-counts", type=int, nargs="+", default=[2, 3],
                        help="Qubit budgets / Bellman-Ford window sizes to sweep for BOTH cg_qaoa and "
                             "cg_bf (same values, see K PINNING).")
    parser.add_argument("--cg-exploration-percents", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--cg-xy-mixer", action="store_true", help="cg_qaoa only.")
    parser.add_argument("--cg-allow-non-improving-columns", action="store_true",
                        help="Also sweep only_improving_columns=False, for both cg_qaoa and cg_bf.")
    parser.add_argument("--cg-max-pricing-nodes", type=int, default=None)
    parser.add_argument("--cg-iterations", type=int, default=ITERATION_CG)
    parser.add_argument("--cg-time-limit", type=int, default=60)

    args = parser.parse_args()
    plot_granularity = "all" if args.plot_all_combos else args.plot_granularity

    run_all_experiment_grid(
        data_dir=args.data_dir,
        num_routes=args.num_routes,
        output_dir=args.output_dir,
        seed=args.seed,
        heuristic_qubit_counts=tuple(args.h_qubit_counts),
        heuristic_exploration_percents=tuple(args.h_exploration_percents),
        heuristic_batch_counts=tuple(args.h_batch_counts),
        heuristic_xy_mixers=(False, True) if args.h_xy_mixer else (False,),
        cg_qubit_counts=tuple(args.cg_qubit_counts),
        cg_exploration_percents=tuple(args.cg_exploration_percents),
        cg_xy_mixers=(False, True) if args.cg_xy_mixer else (False,),
        cg_only_improving_columns_options=(True, False) if args.cg_allow_non_improving_columns else (True,),
        cg_max_pricing_nodes=args.cg_max_pricing_nodes,
        cg_n_iterations=args.cg_iterations,
        cg_time_limit=args.cg_time_limit,
        bf_max_k=args.bf_max_k,
        run_bf_algorithms=not args.no_bf,
        generate_plots=not args.no_plots,
        plot_granularity=plot_granularity,
        plot_formats=tuple(args.plot_formats),
        plot_max_routes=args.plot_max_routes,
    )
