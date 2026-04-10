#!/usr/bin/env python
"""
Build a KDE model of gain/loss rates from PastML ACR output.

Inputs:
  --acr   : pastmlout_marginal.csv  (PastML FLOW/MARGINAL output)
  --tree  : Newick tree used for ACR (Escherichia_coli.nwk)
  --out   : Output pickle path (default: scripts/kde_model.pkl)

The output pickle is a dict:
  {
    "kde"            : scipy.stats.gaussian_kde object  (legacy; no longer used
                       for rate sampling by default),
    "ecoli_total_bl" : float  (IQR-filtered total branch length of ref tree),
    "n_ecoli_taxa"   : int    (number of leaves in the reference tree),
  }

NOTE — primary rate method (as of rate-variant exploration):
  makeSynthData.synth_mutual_4state_nosim now samples rates directly from
  pastmlout_marginal.csv using:
    gain_rate = gains / gain_subsize
    loss_rate = losses / loss_subsize
  and scales via π/λ decomposition (ecoli_total_bl / target_total_bl applied
  to λ only).  This file is still required to produce the ecoli_total_bl and
  n_ecoli_taxa metadata stored in the pickle.  The KDE object itself is only
  used when kde= is explicitly passed to synth_mutual_4state_nosim (legacy).

The KDE is fitted on:
  [log10(gain_rate), log10(loss_rate), root_prob]

where:
  gain_rate = gains_flow / ecoli_total_bl
  loss_rate = losses_flow / ecoli_total_bl

ecoli_total_bl is the IQR-filtered total branch length of the reference tree
(branches above Q3 + 0.5*IQR in log10 space are capped before summing).
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from ete3 import Tree


def compute_ecoli_total_bl(tree_path: str) -> float:
    """Compute IQR-filtered total branch length of the reference E. coli tree.

    Branches above Q3 + 0.5*IQR (in log10 space) are capped at the upper bound
    before summing, consistent with _compute_bl_stats in makeSynthData.py.
    """
    t = Tree(tree_path, format=1)
    branch_lengths = np.array(
        [n.dist for n in t.traverse() if not n.is_root() and n.dist > 0]
    )
    log_bl = np.log10(branch_lengths)
    Q1, Q3 = np.percentile(log_bl, [25, 75])
    IQR = Q3 - Q1
    upper_bound = 10 ** (Q3 + 0.5 * IQR)
    # Cap outlier branches and sum
    capped = np.minimum(branch_lengths, upper_bound)
    return float(capped.sum()), float(upper_bound)


def main():
    parser = argparse.ArgumentParser(description="Build KDE from PastML ACR output.")
    parser.add_argument("--acr",  required=True, help="Path to pastmlout_marginal.csv")
    parser.add_argument("--tree", required=True, help="Path to reference Newick tree (Escherichia_coli.nwk)")
    parser.add_argument("--out",  default="scripts/kde_model.pkl", help="Output pickle path")
    args = parser.parse_args()

    # ── Reference tree branch length and leaf count ───────────────────────────
    print(f"Loading reference tree: {args.tree}")
    ecoli_total_bl, _ = compute_ecoli_total_bl(args.tree)
    print(f"  IQR-filtered total branch length: {ecoli_total_bl:.6f}")

    t = Tree(args.tree, format=1)
    n_ecoli_taxa = len(t.get_leaves())
    print(f"  n_ecoli_taxa (leaves): {n_ecoli_taxa}")

    # ── Load ACR output ───────────────────────────────────────────────────────
    print(f"Loading ACR output: {args.acr}")
    df = pd.read_csv(args.acr)

    required = {"gains_flow", "losses_flow", "root_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ACR CSV: {missing}")

    # ── Compute rates ─────────────────────────────────────────────────────────
    df["gain_rate"] = df["gains_flow"] / ecoli_total_bl
    df["loss_rate"] = df["losses_flow"] / ecoli_total_bl

    # Filter: require positive rates (avoids log(0))
    mask = (df["gain_rate"] > 0) & (df["loss_rate"] > 0)
    df_filt = df[mask].copy()
    print(f"  Genes after filtering (rate > 0): {len(df_filt)} / {len(df)}")

    # ── Fit KDE ───────────────────────────────────────────────────────────────
    log_gain = np.log10(df_filt["gain_rate"].values)
    log_loss = np.log10(df_filt["loss_rate"].values)
    root_prob = df_filt["root_prob"].values

    data = np.stack([log_gain, log_loss, root_prob], axis=0)  # shape (3, N)
    kde = gaussian_kde(data)
    print(f"  KDE fitted on {data.shape[1]} genes, bandwidth: {kde.factor:.4f}")

    # Sanity check: sample from KDE and verify ranges
    test_samples = kde.resample(1000)
    print(f"  Sample log10(gain_rate): mean={test_samples[0].mean():.2f}, "
          f"std={test_samples[0].std():.2f}")
    print(f"  Sample log10(loss_rate): mean={test_samples[1].mean():.2f}, "
          f"std={test_samples[1].std():.2f}")
    print(f"  Sample root_prob: mean={test_samples[2].mean():.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "kde":            kde,
        "ecoli_total_bl": ecoli_total_bl,
        "n_ecoli_taxa":   int(n_ecoli_taxa),
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"KDE saved to {args.out}")


if __name__ == "__main__":
    main()
