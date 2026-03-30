#!/usr/bin/env python
"""
Run SpydrPick on synthetic data and extract results for consecutive trait pairs.

Inputs:
  --systems : Reformatted binary trait CSV (samples × traits, 0/1)
  --outfile : Output CSV path

Pair setup: consecutive pairs (0,1), (2,3), ... matching synthetic data structure.
Classification labels: [0]*3000 + [-1]*300 + [1]*300

SpydrPick computes mutual information (MI) for all position pairs in a FASTA alignment.
We convert binary traits to FASTA (0→A, 1→T), run SpydrPick on all pairs, extract MI
for our consecutive pairs, and compute empirical p-values from the full MI distribution.
Direction is inferred from the observed log-odds ratio (SpydrPick is undirected).
"""

import argparse
import os
import subprocess
import tempfile
import glob

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpydrPick on synthetic trait data.")
    parser.add_argument("--systems", required=True, help="Path to binary trait CSV (samples × traits)")
    parser.add_argument("--outfile", required=True, help="Output CSV path")
    return parser.parse_args()


def csv_to_fasta(data: pd.DataFrame, fasta_path: str):
    """Convert binary (0/1) samples×traits DataFrame to FASTA.

    Each sample becomes a sequence; each trait position maps 0→A, 1→T.
    NaN/missing values map to N (gap category in SpydrPick).
    """
    with open(fasta_path, 'w') as f:
        for sample, row in data.iterrows():
            seq = ''.join(
                'A' if v == 0 else ('T' if v == 1 else 'N')
                for v in row
            )
            f.write(f">{sample}\n{seq}\n")


def compute_log_odds_ratio(col1: np.ndarray, col2: np.ndarray) -> float:
    """Compute log-odds ratio for two binary vectors.

    LOR > 0 → co-occurrence (positive association)
    LOR < 0 → co-exclusion (negative association)
    Returns 0.0 if any marginal count is zero (undefined).
    """
    n11 = np.sum((col1 == 1) & (col2 == 1))
    n10 = np.sum((col1 == 1) & (col2 == 0))
    n01 = np.sum((col1 == 0) & (col2 == 1))
    n00 = np.sum((col1 == 0) & (col2 == 0))
    if n10 == 0 or n01 == 0 or n11 == 0 or n00 == 0:
        return 0.0
    return float(np.log((n11 * n00) / (n10 * n01)))


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    data = pd.read_csv(args.systems, index_col=0)
    n_traits = data.shape[1]
    n_pairs_total = n_traits * (n_traits - 1) // 2

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "traits.fasta")
        csv_to_fasta(data, fasta_path)

        # Run SpydrPick on all pairs
        # --ld-threshold 0    : include all pairs regardless of positional distance
        # --mi-values         : output ALL pair MI values for empirical p-value computation
        # --linear-genome     : traits are not on a circular chromosome
        # --maf-threshold 0   : do not filter any traits based on frequency
        # --no-aracne         : skip ARACNE post-processing (we want raw MI for all pairs)
        cmd = [
            "SpydrPick",
            "-v", fasta_path,
            "--ld-threshold", "0",
            "--mi-values", str(n_pairs_total),
            "--linear-genome",
            "--maf-threshold", "0",
            "--no-aracne",
        ]
        result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SpydrPick failed:\n{result.stderr}")

        # Locate output file: *.spydrpick_couplings.*edges
        edge_files = glob.glob(os.path.join(tmpdir, "*.spydrpick_couplings.*edges"))
        if not edge_files:
            raise FileNotFoundError(
                f"SpydrPick edges output not found in {tmpdir}. "
                f"SpydrPick stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        edge_file = edge_files[0]

        # Parse output: whitespace-delimited columns pos1 pos2 genome_distance ARACNE MI
        # Positions are 1-based. Stream through file to build:
        #   1. mi_lookup: {(pos1, pos2): MI} for consecutive pairs
        #   2. all_mi: list of all MI values for empirical distribution
        consecutive_set = set()
        for i in range(0, n_traits, 2):
            consecutive_set.add((i + 1, i + 2))  # 1-based

        mi_lookup = {}
        all_mi = []

        with open(edge_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                pos1, pos2 = int(parts[0]), int(parts[1])
                mi = float(parts[4])
                all_mi.append(mi)
                key = (min(pos1, pos2), max(pos1, pos2))
                if key in consecutive_set:
                    mi_lookup[key] = mi

    all_mi = np.array(all_mi)
    n_total = len(all_mi)

    # Build results for consecutive pairs
    n_consecutive = n_traits // 2
    labels = [0] * 3000 + [-1] * 300 + [1] * 300
    if n_consecutive != len(labels):
        raise ValueError(
            f"Expected {len(labels)} consecutive pairs, got {n_consecutive}. "
            "Check synthetic data structure (should be 3600 traits → 1800 pairs)."
        )

    cols = list(data.columns)
    rows = []
    for idx in range(n_consecutive):
        i = idx * 2
        trait1 = cols[i]
        trait2 = cols[i + 1]
        pos1, pos2 = i + 1, i + 2  # 1-based

        key = (min(pos1, pos2), max(pos1, pos2))
        mi_val = mi_lookup.get(key, np.nan)

        if np.isnan(mi_val):
            # Pair not in SpydrPick output → assign worst p-value
            p_value = 1.0
            direction = 0
        else:
            # Empirical p-value: fraction of all pairs with MI >= this value
            p_value = float(np.sum(all_mi >= mi_val) / n_total)
            lor = compute_log_odds_ratio(
                data.iloc[:, i].values,
                data.iloc[:, i + 1].values,
            )
            direction = 1 if lor > 0 else -1

        rows.append({
            'trait1':    trait1,
            'trait2':    trait2,
            'mi_score':  mi_val,
            'p_value':   p_value,
            'direction': direction,
            'label':     labels[idx],
        })

    res = pd.DataFrame(rows)
    res.to_csv(args.outfile, index=False)
    print(f"Saved SpydrPick results to {args.outfile}")


if __name__ == "__main__":
    main()
