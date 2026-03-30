#!/usr/bin/env python
"""
Run Coinfinder on synthetic data and extract results for consecutive trait pairs.

Inputs:
  --systems     : Reformatted binary trait CSV (samples × traits, 0/1)
  --tree        : Reformatted Newick tree file
  --outfile     : Output CSV path
  --threads     : Number of CPU cores to pass to coinfinder per batch (default: 8)
  --batch-size  : Number of traits per coinfinder batch (default: 100 = 50 pairs)

Pair setup: consecutive pairs (0,1), (2,3), ... matching synthetic data structure.
Classification labels: [0]*3000 + [-1]*300 + [1]*300

Strategy: run coinfinder in non-overlapping batches of BATCH_SIZE consecutive traits.
Running all 7200 traits at once is infeasible due to O(n²) pair computation (~days).
Batches of ~100 traits give coinfinder enough gene-frequency background for the
phylogenetic Beta correction to work, while keeping each run to ~3 minutes with
multiple threads. 36 batches × ~3 min ≈ ~1.7 hours per tree/effect-size combo.

Per-pair runs (2 traits at a time) do NOT work — the Beta correction degenerates
to p=1.0 for all pairs without enough background genes.

Coinfinder is run twice per batch:
  1. Associate mode (-a): co-occurring pairs    → direction = +1
  2. Dissociate mode (-d): counter pairs        → direction = -1

Output: _pairs.tsv with columns Source, Target, p (and others).
No multiple-testing correction (-n): raw p-values, default 0.05 threshold.
Consecutive pairs absent from the output are assigned p_value=1.0 / direction=0.

Coinfinder input (long/edge format, no -I flag):
  Two tab-separated columns: gene_name<TAB>genome_name
  One row per (gene, genome) where the gene is present (value == 1).
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd


BATCH_SIZE_DEFAULT = 100  # traits per batch (50 consecutive pairs per batch)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Coinfinder on synthetic trait data.")
    parser.add_argument("--systems",    required=True, help="Path to binary trait CSV (samples × traits)")
    parser.add_argument("--tree",       required=True, help="Path to Newick tree file")
    parser.add_argument("--outfile",    required=True, help="Output CSV path")
    parser.add_argument("--threads",    type=int, default=8, help="Coinfinder CPU cores per batch")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
                        help="Number of traits per coinfinder batch (default: 100)")
    return parser.parse_args()


def build_long_format(data: pd.DataFrame, path: str):
    """Write gene-genome edge file: 'trait<TAB>sample' per present cell.

    Uses vectorized stack() + boolean filter — avoids slow nested Python loops.
    """
    present = data.stack()
    present = present[present == 1]
    with open(path, 'w') as f:
        for (sample, trait) in present.index:
            f.write(f"{trait}\t{sample}\n")


def run_coinfinder_mode(long_input: str, tree: str, mode_flag: str,
                        prefix: str, threads: int) -> int:
    """Run coinfinder in associate (-a) or dissociate (-d) mode."""
    cmd = [
        "coinfinder",
        "-i", long_input,
        "-p", os.path.abspath(tree),   # phylogeny flag (-p, NOT -a)
        mode_flag,                      # -a (associate) or -d (dissociate)
        "-o", prefix,
        "-n",                           # no multiple-testing correction (raw p-values)
        "-x", str(threads),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Coinfinder ({mode_flag}) stderr: {result.stderr[:300]}", flush=True)
    return result.returncode


def parse_pairs_tsv(path: str) -> dict:
    """Parse coinfinder _pairs.tsv → {frozenset({t1, t2}): p_value}."""
    lookup = {}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return lookup
    df = pd.read_csv(path, sep='\t')
    if not {'Source', 'Target', 'p'}.issubset(df.columns):
        print(f"  Warning: unexpected columns in {path}: {list(df.columns)}", flush=True)
        return lookup
    for _, row in df.iterrows():
        key = frozenset({str(row['Source']), str(row['Target'])})
        lookup[key] = float(row['p'])
    return lookup


def run_batch(batch_data: pd.DataFrame, tree: str, threads: int,
              tmpdir: str, batch_idx: int) -> tuple[dict, dict]:
    """Run one coinfinder batch; return (assoc_lookup, dissoc_lookup)."""
    edge_file = os.path.join(tmpdir, f"edges_{batch_idx}.tsv")
    build_long_format(batch_data, edge_file)

    assoc_prefix  = os.path.join(tmpdir, f"assoc_{batch_idx}")
    dissoc_prefix = os.path.join(tmpdir, f"dissoc_{batch_idx}")

    run_coinfinder_mode(edge_file, tree, "-a", assoc_prefix,  threads)
    run_coinfinder_mode(edge_file, tree, "-d", dissoc_prefix, threads)

    assoc  = parse_pairs_tsv(f"{assoc_prefix}_pairs.tsv")
    dissoc = parse_pairs_tsv(f"{dissoc_prefix}_pairs.tsv")
    return assoc, dissoc


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    data = pd.read_csv(args.systems, index_col=0)
    cols     = list(data.columns)
    n_traits = len(cols)
    n_consecutive = n_traits // 2

    labels = [0] * 3000 + [-1] * 300 + [1] * 300
    if n_consecutive != len(labels):
        raise ValueError(
            f"Expected {len(labels)} consecutive pairs, got {n_consecutive}. "
            "Check synthetic data structure (should be 3600 traits → 1800 pairs)."
        )

    batch_size = args.batch_size
    # Ensure batch_size is even so each batch contains only complete consecutive pairs
    if batch_size % 2 != 0:
        batch_size += 1

    n_batches = (n_traits + batch_size - 1) // batch_size
    tree = args.tree
    print(f"Running coinfinder on {n_consecutive} pairs in {n_batches} batches "
          f"of {batch_size} traits ({batch_size//2} pairs each) "
          f"with {args.threads} threads...", flush=True)

    # Accumulate lookups from all batches
    assoc_all  = {}
    dissoc_all = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for b in range(n_batches):
            start = b * batch_size
            end   = min(start + batch_size, n_traits)
            batch_cols = cols[start:end]
            batch_data = data[batch_cols]

            n_batch_pairs = len(batch_cols) * (len(batch_cols) - 1) // 2
            print(f"  Batch {b+1}/{n_batches}: traits {start}-{end-1} "
                  f"({len(batch_cols)} traits, {n_batch_pairs} pairs)...", flush=True)

            assoc, dissoc = run_batch(batch_data, tree, args.threads, tmpdir, b)
            print(f"    assoc={len(assoc)} sig, dissoc={len(dissoc)} sig", flush=True)

            assoc_all.update(assoc)
            dissoc_all.update(dissoc)

    rows = []
    for idx in range(n_consecutive):
        i = idx * 2
        trait1 = cols[i]
        trait2 = cols[i + 1]
        key = frozenset({trait1, trait2})

        p_assoc  = assoc_all.get(key, np.nan)
        p_dissoc = dissoc_all.get(key, np.nan)

        both_missing = np.isnan(p_assoc) and np.isnan(p_dissoc)
        if both_missing:
            p_value   = 1.0
            direction = 0
        elif np.isnan(p_dissoc) or (not np.isnan(p_assoc) and p_assoc <= p_dissoc):
            p_value   = float(p_assoc)
            direction = 1
        else:
            p_value   = float(p_dissoc)
            direction = -1

        rows.append({
            "trait1":    trait1,
            "trait2":    trait2,
            "p_value":   p_value,
            "direction": direction,
            "label":     labels[idx],
        })

    res = pd.DataFrame(rows)
    res.to_csv(args.outfile, index=False)
    print(f"Saved Coinfinder results to {args.outfile}")
    n_assoc  = sum(1 for r in rows if r["direction"] ==  1)
    n_dissoc = sum(1 for r in rows if r["direction"] == -1)
    n_null   = sum(1 for r in rows if r["direction"] ==  0)
    print(f"  associate={n_assoc}, dissociate={n_dissoc}, null/missing={n_null}")


if __name__ == "__main__":
    main()
