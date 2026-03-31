#!/usr/bin/env python
"""
Run SpydrPick on synthetic data and evaluate against known pair labels.

Inputs:
  --systems     : Reformatted binary trait CSV (samples × traits, 0/1)
  --pair_labels : CSV with columns trait1, trait2, direction (from generateData)
  --outfile     : Output CSV path

SpydrPick computes mutual information (MI) for all position pairs in a FASTA
alignment.  We convert binary traits to FASTA (0→A, 1→T), run SpydrPick on
all pairs with --no-filter-alignment to disable position-based filtering
(appropriate since our FASTA columns are synthetic binary traits with no
meaningful genomic coordinates), extract MI for each labeled pair, and compute
empirical p-values from the full MI distribution.

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
    parser.add_argument("--systems",     required=True, help="Path to binary trait CSV (samples × traits)")
    parser.add_argument("--pair_labels", required=True, help="Path to pair_labels.csv")
    parser.add_argument("--outfile",     required=True, help="Output CSV path")
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

    data        = pd.read_csv(args.systems, index_col=0)
    pair_labels = pd.read_csv(args.pair_labels)
    n_traits    = data.shape[1]
    n_pairs_total = n_traits * (n_traits - 1) // 2

    cols = list(data.columns)
    col_to_pos = {col: i + 1 for i, col in enumerate(cols)}  # 1-based positions

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "traits.fasta")
        csv_to_fasta(data, fasta_path)

        # --no-filter-alignment : disable all position-based filtering
        #   (our FASTA columns have no meaningful genomic coordinates)
        # --mi-values           : output ALL pair MI values for empirical p-value
        # --maf-threshold 0     : do not filter traits by minor allele frequency
        # --no-aracne           : skip ARACNE post-processing (want raw MI)
        cmd = [
            "SpydrPick",
            "-v", fasta_path,
            "--no-filter-alignment",
            "--mi-values", str(n_pairs_total),
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

        # Build labeled-pair lookup set for efficient lookup
        labeled_pairs = set()
        for _, row in pair_labels.iterrows():
            t1, t2 = str(row["trait1"]), str(row["trait2"])
            if t1 in col_to_pos and t2 in col_to_pos:
                p1, p2 = col_to_pos[t1], col_to_pos[t2]
                labeled_pairs.add((min(p1, p2), max(p1, p2)))

        # Parse edges: columns are pos1 pos2 genome_distance ARACNE MI
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
                if key in labeled_pairs:
                    mi_lookup[key] = mi

    all_mi = np.array(all_mi)
    n_total = len(all_mi)
    print(f"SpydrPick: {n_total} total pairs computed, "
          f"{len(mi_lookup)} labeled pairs found in output.", flush=True)

    rows = []
    for _, pl_row in pair_labels.iterrows():
        trait1    = str(pl_row["trait1"])
        trait2    = str(pl_row["trait2"])
        label     = int(pl_row["direction"])

        if trait1 not in col_to_pos or trait2 not in col_to_pos:
            rows.append({"trait1": trait1, "trait2": trait2,
                         "mi_score": np.nan, "p_value": 1.0,
                         "direction": 0, "label": label})
            continue

        p1, p2 = col_to_pos[trait1], col_to_pos[trait2]
        key    = (min(p1, p2), max(p1, p2))
        mi_val = mi_lookup.get(key, np.nan)

        if np.isnan(mi_val) or n_total == 0:
            p_value   = 1.0
            direction = 0
        else:
            p_value = float(np.sum(all_mi >= mi_val) / n_total)
            lor = compute_log_odds_ratio(
                data[trait1].values,
                data[trait2].values,
            )
            direction = 1 if lor > 0 else -1

        rows.append({
            "trait1":    trait1,
            "trait2":    trait2,
            "mi_score":  mi_val,
            "p_value":   p_value,
            "direction": direction,
            "label":     label,
        })

    res = pd.DataFrame(rows)
    res.to_csv(args.outfile, index=False)
    print(f"Saved SpydrPick results to {args.outfile}")


if __name__ == "__main__":
    main()
