#!/usr/bin/env python
"""
Run Coinfinder on synthetic data and evaluate against known pair labels.

Inputs:
  --systems     : Reformatted binary trait CSV (samples × traits, 0/1)
  --tree        : Reformatted Newick tree file
  --pair_labels : CSV with columns trait1, trait2, direction (from generateData)
  --outfile     : Output CSV path
  --threads     : Number of CPU cores to pass to coinfinder (default: 8)

Coinfinder is run twice on the full trait matrix:
  1. Associate mode (-a): co-occurring pairs    → direction = +1
  2. Dissociate mode (-d): counter pairs        → direction = -1

Output: one row per pair in pair_labels with p_value and inferred direction.
Pairs absent from Coinfinder output are assigned p_value=1.0 / direction=0.

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Coinfinder on synthetic trait data.")
    parser.add_argument("--systems",     required=True, help="Path to binary trait CSV (samples × traits)")
    parser.add_argument("--tree",        required=True, help="Path to Newick tree file")
    parser.add_argument("--pair_labels", required=True, help="Path to pair_labels.csv")
    parser.add_argument("--outfile",     required=True, help="Output CSV path")
    parser.add_argument("--threads",     type=int, default=8, help="Coinfinder CPU cores")
    return parser.parse_args()


def build_long_format(data: pd.DataFrame, path: str):
    """Write gene-genome edge file: 'trait<TAB>sample' per present cell."""
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
        "-p", os.path.abspath(tree),
        mode_flag,
        "-o", prefix,
        "-n",                    # no multiple-testing correction (raw p-values)
        "-x", str(threads),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Coinfinder ({mode_flag}) stderr: {result.stderr[:500]}", flush=True)
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


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    data        = pd.read_csv(args.systems, index_col=0)
    pair_labels = pd.read_csv(args.pair_labels)

    print(f"Running Coinfinder on {data.shape[1]} traits x {data.shape[0]} samples...",
          flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        edge_file     = os.path.join(tmpdir, "edges.tsv")
        assoc_prefix  = os.path.join(tmpdir, "assoc")
        dissoc_prefix = os.path.join(tmpdir, "dissoc")

        build_long_format(data, edge_file)

        run_coinfinder_mode(edge_file, args.tree, "-a", assoc_prefix,  args.threads)
        run_coinfinder_mode(edge_file, args.tree, "-d", dissoc_prefix, args.threads)

        assoc  = parse_pairs_tsv(f"{assoc_prefix}_pairs.tsv")
        dissoc = parse_pairs_tsv(f"{dissoc_prefix}_pairs.tsv")

    print(f"  Associate significant pairs:  {len(assoc)}", flush=True)
    print(f"  Dissociate significant pairs: {len(dissoc)}", flush=True)

    rows = []
    for _, pl_row in pair_labels.iterrows():
        trait1 = str(pl_row["trait1"])
        trait2 = str(pl_row["trait2"])
        label  = int(pl_row["direction"])
        key    = frozenset({trait1, trait2})

        p_assoc  = assoc.get(key, np.nan)
        p_dissoc = dissoc.get(key, np.nan)

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
            "label":     label,
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
