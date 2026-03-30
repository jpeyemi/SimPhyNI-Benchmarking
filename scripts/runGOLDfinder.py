#!/usr/bin/env python
"""
Run GOLDfinder on synthetic data and extract results for consecutive trait pairs.

Inputs:
  --systems : Reformatted binary trait CSV (samples × traits, 0/1)
  --tree    : Reformatted Newick tree
  --outfile : Output CSV path

Pair setup: consecutive pairs (0,1), (2,3), ... matching synthetic data structure.
Classification labels: [0]*3000 + [-1]*300 + [1]*300

GOLDfinder computes phylogenetically-informed co-occurrence (association) and
co-exclusion (dissociation) scores for all trait pairs using a branch-scoring method.
We run GOLDfinder in batches of BATCH_SIZE consecutive pairs to avoid OOM when
processing all 25M+ pairs at once. Each batch output is small and parsed directly.
All (score, batch) combinations are run in parallel using ThreadPoolExecutor.
"""

import argparse
import copy
import os
import subprocess
import sys
import tempfile
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


SCORES = ['terminal', 'simultaneous', 'subsequent', 'coinfinder']

# Number of consecutive pairs per GOLDfinder batch (= 2*BATCH_SIZE traits per run).
# With BATCH_SIZE=250: 500 traits → ~125K pairs per run, output file ~10 MB.
BATCH_SIZE = 250


def parse_args():
    parser = argparse.ArgumentParser(description="Run GOLDfinder on synthetic trait data.")
    parser.add_argument("--systems", required=True, help="Path to binary trait CSV (samples × traits)")
    parser.add_argument("--tree",    required=True, help="Path to Newick tree file")
    parser.add_argument("--outdir",  required=True, help="Output directory (one CSV per score)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel GOLDfinder processes (default: all CPUs)")
    return parser.parse_args()


GOLDFINDER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "goldfinder",
    "goldfinder", "goldfinder.py"
)


def run_goldfinder(matrix_path: str, tree_path: str, outdir: str, score: str = "terminal"):
    """Run GOLDfinder from the cloned repo at <project_root>/goldfinder/goldfinder.py."""
    base_cmd = [
        "-i", matrix_path,
        "-t", tree_path,
        "-f", "matrix",   # genes-as-rows binary CSV
        "-c", "both",     # association and dissociation
        "-s", score,      # scoring method (terminal matches htreewas terminal)
        "-o", outdir,
        "-a", "1.0",      # significance threshold of 1.0
        "-pcor", "none",  # no p-value correction
        "-n",             # no clustering (expensive and unneeded)
        "--seed 42"
    ]

    # Cap internal threading so parallel workers don't over-subscribe CPUs.
    # The simultaneous score uses Numba prange; terminal/subsequent/coinfinder are serial.
    env = copy.copy(os.environ)
    env["NUMBA_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    result = subprocess.run(
        [sys.executable, GOLDFINDER_SCRIPT] + base_cmd,
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"GOLDfinder failed:\n{result.stderr}")
    return result


def parse_significant_pairs(filepath: str) -> dict:
    """Parse a GOLDfinder significant pairs CSV into a lookup dict.

    Returns {(trait1, trait2): p_value} using unordered (frozenset-like) keys
    so pair lookup works regardless of column order in the output file.
    """
    if not os.path.exists(filepath):
        return {}

    df = pd.read_csv(filepath)
    if df.empty:
        return {}

    # Detect gene/trait name columns (first two non-numeric-looking columns)
    id_cols = [c for c in df.columns if df[c].dtype == object][:2]
    if len(id_cols) < 2:
        id_cols = list(df.columns[:2])

    # Detect p-value column
    pval_col = None
    for candidate in ['p_value', 'pvalue', 'p-value', 'pval', 'P_value', 'P-value']:
        if candidate in df.columns:
            pval_col = candidate
            break
    if pval_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            pval_col = numeric_cols[-1]
        else:
            return {}

    lookup = {}
    for _, row in df.iterrows():
        t1 = str(row[id_cols[0]])
        t2 = str(row[id_cols[1]])
        key = tuple(sorted([t1, t2]))
        lookup[key] = float(row[pval_col])

    return lookup


def build_results(data, assoc_lookup, dissoc_lookup, labels):
    """Build consecutive-pair result rows from association/dissociation lookups."""
    cols = list(data.columns)
    n_consecutive = len(cols) // 2
    rows = []
    for idx in range(n_consecutive):
        i = idx * 2
        trait1 = cols[i]
        trait2 = cols[i + 1]
        key = tuple(sorted([trait1, trait2]))

        if key in assoc_lookup and key in dissoc_lookup:
            if assoc_lookup[key] <= dissoc_lookup[key]:
                p_value, direction = assoc_lookup[key], 1
            else:
                p_value, direction = dissoc_lookup[key], -1
        elif key in assoc_lookup:
            p_value, direction = assoc_lookup[key], 1
        elif key in dissoc_lookup:
            p_value, direction = dissoc_lookup[key], -1
        else:
            p_value, direction = 1.0, 0

        rows.append({
            'trait1':    trait1,
            'trait2':    trait2,
            'p_value':   p_value,
            'direction': direction,
            'label':     labels[idx],
        })
    return rows


def run_batch(score, batch_start, batch_end, cols, data, tree_path, tmpdir, labels):
    """Run one (score, batch) GOLDfinder job; returns (score, batch_start, rows)."""
    batch_traits = []
    for idx in range(batch_start, batch_end):
        batch_traits.extend([cols[idx * 2], cols[idx * 2 + 1]])

    batch_data = data[batch_traits]
    matrix_path = os.path.join(tmpdir, f"matrix_{score}_{batch_start}.csv")
    batch_data.T.to_csv(matrix_path)

    gf_outdir = os.path.join(tmpdir, f"gf_{score}_{batch_start}")
    os.makedirs(gf_outdir, exist_ok=True)
    run_goldfinder(matrix_path, tree_path, gf_outdir, score=score)

    assoc_files = glob.glob(os.path.join(gf_outdir, "*association*significant*pairs*.csv"))
    dissoc_files = glob.glob(os.path.join(gf_outdir, "*dissociation*significant*pairs*.csv"))

    assoc_lookup = {}
    for f in assoc_files:
        assoc_lookup.update(parse_significant_pairs(f))

    dissoc_lookup = {}
    for f in dissoc_files:
        dissoc_lookup.update(parse_significant_pairs(f))

    batch_labels = labels[batch_start:batch_end]
    rows = build_results(batch_data, assoc_lookup, dissoc_lookup, batch_labels)
    print(f"  [{score}] batch {batch_start}–{batch_end-1}: {len(rows)} pairs processed",
          flush=True)
    return score, batch_start, rows


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = pd.read_csv(args.systems, index_col=0)
    cols = list(data.columns)
    n_traits = len(cols)
    n_consecutive = n_traits // 2

    labels = [0] * 3000 + [-1] * 300 + [1] * 300
    if n_consecutive != len(labels):
        raise ValueError(
            f"Expected {len(labels)} consecutive pairs, got {n_consecutive}. "
            "Check synthetic data structure (should be 3600 traits → 1800 pairs)."
        )

    # All (score, batch) combinations are independent — run them in parallel.
    tasks = [
        (score, batch_start, min(batch_start + BATCH_SIZE, n_consecutive))
        for score in SCORES
        for batch_start in range(0, n_consecutive, BATCH_SIZE)
    ]
    print(f"Submitting {len(tasks)} GOLDfinder jobs with {args.workers} workers")

    all_rows = {score: {} for score in SCORES}  # score → {batch_start: rows}

    with tempfile.TemporaryDirectory() as tmpdir:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_batch, score, bs, be, cols, data, args.tree, tmpdir, labels):
                    (score, bs)
                for score, bs, be in tasks
            }
            for future in as_completed(futures):
                score, batch_start, rows = future.result()
                all_rows[score][batch_start] = rows

    for score in SCORES:
        ordered = []
        for bs in sorted(all_rows[score]):
            ordered.extend(all_rows[score][bs])
        outfile = os.path.join(args.outdir, f"goldfinder_{score}_results.csv")
        pd.DataFrame(ordered).to_csv(outfile, index=False)
        print(f"Saved GOLDfinder ({score}) results to {outfile}")


if __name__ == "__main__":
    main()
