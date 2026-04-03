"""
Recompute D-statistics in-place for all pair_labels.csv files under a root folder.

For each pair_labels.csv the script:
  1. Locates the matching synthetic_data.csv (same directory) and tree file.
  2. Loads the tree, precomputes BM vectors once per tree.
  3. For every row, reads tip-state vectors from synthetic_data.csv, computes
     prevalence-aware (d1, d2, d_statistic, d_stratum) and overwrites those
     four columns.  All other columns are left untouched.

Tree discovery (in order of precedence):
  - <dir>/tree.nwk
  - <dir>/reformated_tree.nwk
  - <dir>/../tree.nwk          (one level up, for es{N} sub-directories)
  - <dir>/../../tree.nwk       (two levels up, same)

Usage
-----
    python recompute_d_statistics.py <root_folder> [options]

Options
-------
    --n-permutations INT   Number of BM/permutation samples (default: 999)
    --d-low FLOAT          D ≤ this → low_independence   (default: -0.05)
    --d-high FLOAT         D > this → high_independence  (default:  0.05)
    --dry-run              Print what would be done without writing files
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree

# Allow running from any working directory as long as scripts/ is on the path
_SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPTS_DIR))

from d_statistic import get_or_calibrate, get_null_distributions, compute_d_statistic


# ── Tree discovery ────────────────────────────────────────────────────────────

_TREE_CANDIDATES = [
    "tree.nwk",
    "reformated_tree.nwk",
    "../tree.nwk",
    "../../tree.nwk",
    "../reformated_tree.nwk",
]


def find_tree(data_dir: Path) -> Path | None:
    for rel in _TREE_CANDIDATES:
        candidate = (data_dir / rel).resolve()
        if candidate.is_file():
            return candidate
    return None


# ── D-statistic helpers ───────────────────────────────────────────────────────

def assign_stratum(d: float, d_low: float, d_high: float) -> str:
    if np.isnan(d):       return "degenerate"
    if d <= d_low:        return "low_independence"
    if d > d_high:        return "high_independence"
    return "mid_independence"


def recompute_pair_labels(
    pair_labels_path: Path,
    n_permutations: int,
    d_low: float,
    d_high: float,
    dry_run: bool,
) -> bool:
    """
    Recompute d1, d2, d_statistic, d_stratum for every row in pair_labels_path.
    Returns True on success, False on skip/error.
    """
    data_dir = pair_labels_path.parent

    # ── Locate synthetic data ─────────────────────────────────────────────────
    synth_path = data_dir / "synthetic_data.csv"
    if not synth_path.is_file():
        print(f"  SKIP  no synthetic_data.csv in {data_dir}")
        return False

    # ── Locate tree ───────────────────────────────────────────────────────────
    tree_path = find_tree(data_dir)
    if tree_path is None:
        print(f"  SKIP  no tree file found near {data_dir}")
        return False

    print(f"  tree  {tree_path}")

    if dry_run:
        print(f"  DRY   would recompute {pair_labels_path}")
        return True

    # ── Load data ─────────────────────────────────────────────────────────────
    pl    = pd.read_csv(pair_labels_path)
    synth = pd.read_csv(synth_path, index_col=0)   # rows=leaves, cols=traits

    t = Tree(str(tree_path), format=1)

    # ── Precompute BM vectors once per tree (cached across calls) ─────────────
    tree_structure, bm_leaf_vals = get_or_calibrate(
        str(tree_path), t, n_permutations=n_permutations
    )
    ts_leaf_names = tree_structure["leaf_names"]

    # leaf_order maps leaf name → row index in synth (which is indexed by leaf)
    leaf_order = {name: i for i, name in enumerate(synth.index)}

    # ── Recompute per row ─────────────────────────────────────────────────────
    new_d1, new_d2, new_d_stat, new_stratum = [], [], [], []

    missing_traits = set()

    for _, row in pl.iterrows():
        t1_name, t2_name = row["trait1"], row["trait2"]

        # Pull tip-state columns from synth, reordered to match tree leaf order
        def get_ordered(trait_name):
            if trait_name not in synth.columns:
                missing_traits.add(trait_name)
                return None
            col = synth[trait_name].values.astype(float)
            return np.array([col[leaf_order[n]] for n in ts_leaf_names], dtype=float)

        ord1 = get_ordered(t1_name)
        ord2 = get_ordered(t2_name)

        if ord1 is None or ord2 is None:
            new_d1.append(float("nan"))
            new_d2.append(float("nan"))
            new_d_stat.append(float("nan"))
            new_stratum.append("degenerate")
            continue

        rm1, bm1 = get_null_distributions(
            str(tree_path), tree_structure, bm_leaf_vals, ord1.mean(), n_permutations
        )
        rm2, bm2 = get_null_distributions(
            str(tree_path), tree_structure, bm_leaf_vals, ord2.mean(), n_permutations
        )

        d1 = compute_d_statistic(tree_structure, ord1, rm1, bm1)
        d2 = compute_d_statistic(tree_structure, ord2, rm2, bm2)
        finite = [v for v in (d1, d2) if not np.isnan(v)]
        d_pair = float(max(finite)) if finite else float("nan")

        new_d1.append(d1)
        new_d2.append(d2)
        new_d_stat.append(d_pair)
        new_stratum.append(assign_stratum(d_pair, d_low, d_high))

    if missing_traits:
        print(f"  WARN  {len(missing_traits)} traits not found in synthetic_data: "
              f"{sorted(missing_traits)[:5]}{'...' if len(missing_traits) > 5 else ''}")

    # ── Overwrite the four D columns ──────────────────────────────────────────
    pl["d1"]          = new_d1
    pl["d2"]          = new_d2
    pl["d_statistic"] = new_d_stat
    pl["d_stratum"]   = new_stratum

    pl.to_csv(pair_labels_path, index=False)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Recompute prevalence-aware D-statistics in pair_labels.csv files."
    )
    parser.add_argument("root", help="Root folder to search recursively")
    parser.add_argument("--n-permutations", type=int, default=999,
                        metavar="INT")
    parser.add_argument("--d-low",  type=float, default=-0.05, metavar="FLOAT")
    parser.add_argument("--d-high", type=float, default=0.05,  metavar="FLOAT")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Error: {root} is not a directory")

    pair_label_files = sorted(root.rglob("pair_labels.csv"))
    if not pair_label_files:
        sys.exit(f"No pair_labels.csv files found under {root}")

    print(f"Found {len(pair_label_files)} pair_labels.csv file(s) under {root}\n")

    ok = skipped = 0
    for plf in pair_label_files:
        print(f"[{ok + skipped + 1}/{len(pair_label_files)}] {plf.relative_to(root)}")
        success = recompute_pair_labels(
            plf,
            n_permutations=args.n_permutations,
            d_low=args.d_low,
            d_high=args.d_high,
            dry_run=args.dry_run,
        )
        if success:
            ok += 1
            if not args.dry_run:
                print(f"  DONE  wrote {plf}")
        else:
            skipped += 1
        print()

    print(f"Finished: {ok} updated, {skipped} skipped.")


if __name__ == "__main__":
    main()
