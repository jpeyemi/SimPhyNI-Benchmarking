#!/usr/bin/env python
"""
Generate a synthetic dataset (tree + binary trait matrix) for runtime benchmarking.

Usage:
  python generate_data.py --n_leaves 500 --n_traits 100 \
      --tree_out data/500_100_rep0/tree.nwk \
      --traits_out data/500_100_rep0/traits.csv \
      --seed 0
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

# Allow importing generateTree from the parent scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from generateTree import generate_msprime_tree


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_leaves',   type=int, required=True)
    p.add_argument('--n_traits',   type=int, required=True)
    p.add_argument('--tree_out',   required=True)
    p.add_argument('--traits_out', required=True)
    p.add_argument('--seed',       type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.dirname(args.tree_out),   exist_ok=True)
    os.makedirs(os.path.dirname(args.traits_out), exist_ok=True)

    # ── Tree ──────────────────────────────────────────────────────────────────
    tree = generate_msprime_tree(target_leaves=args.n_leaves)
    leaf_names = [leaf.name for leaf in tree.get_leaves()]
    tree.write(outfile=args.tree_out, format=1)
    print(f"Tree written: {len(leaf_names)} leaves → {args.tree_out}")

    # ── Traits ────────────────────────────────────────────────────────────────
    # iid Bernoulli(0.5) — no evolutionary signal needed for runtime benchmarking
    trait_matrix = rng.integers(0, 2, size=(len(leaf_names), args.n_traits))
    cols = [f"trait_{i}" for i in range(args.n_traits)]
    df = pd.DataFrame(trait_matrix, index=leaf_names, columns=cols)
    df.index.name = 'Sample'
    df.to_csv(args.traits_out)
    print(f"Traits written: {df.shape} → {args.traits_out}")


if __name__ == '__main__':
    main()
