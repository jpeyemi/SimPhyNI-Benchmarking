#!/usr/bin/env python
"""
Build a KDE model of gain/loss rates from PastML ACR output.

Inputs:
  --acr   : pastmlout_marginal.csv  (PastML FLOW/MARGINAL output)
  --tree  : Newick tree used for ACR (Escherichia_coli.nwk)
  --out   : Output pickle path (default: scripts/kde_model.pkl)

The KDE is fitted on:
  [log10(gain_rate), log10(loss_rate), root_prob]

where:
  gain_rate = gains_flow / gain_subsize_marginal_thresh
  loss_rate = losses_flow / loss_subsize_marginal_thresh

gain_subsize_marginal_thresh is the sum of subtree branch lengths below
each gain event, with IQR-outlier branches scaled down. Dividing
gains_flow by this gives the gain rate per unit of IQR-filtered subtree
evolutionary space.

The output pickle is a dict:
  {
    "kde"              : scipy.stats.gaussian_kde object,
    "ecoli_total_bl"   : float  (IQR-filtered total branch length of ref tree),
    "ecoli_mean_subsize": float (mean gain_subsize_marginal_thresh across genes),
  }

Rate scaling for new trees (in makeSynthData.py):
  gain_rates = kde_sample_gains * ecoli_mean_subsize / new_tree_total_bl

This preserves the expected total transition count:
  kde_sample_gains * ecoli_mean_subsize  ≈  expected gains on E. coli tree
  / new_tree_total_bl                    →  rate per new-tree branch-length unit
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

    # ── Reference tree branch length ──────────────────────────────────────────
    print(f"Loading reference tree: {args.tree}")
    ecoli_total_bl, _ = compute_ecoli_total_bl(args.tree)
    print(f"  IQR-filtered total branch length: {ecoli_total_bl:.6f}")

    # ── Load ACR output ───────────────────────────────────────────────────────
    print(f"Loading ACR output: {args.acr}")
    df = pd.read_csv(args.acr)

    required = {"gains_flow", "losses_flow",
                "gain_subsize_marginal_thresh", "loss_subsize_marginal_thresh",
                "root_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ACR CSV: {missing}")

    # ── Compute rates ─────────────────────────────────────────────────────────
    df["gain_rate"] = df["gains_flow"] / df["gain_subsize_marginal_thresh"]
    df["loss_rate"] = df["losses_flow"] / df["loss_subsize_marginal_thresh"]

    # Filter: require positive rates and positive subsize (avoids log(0))
    mask = (
        (df["gain_rate"] > 0) &
        (df["loss_rate"] > 0) &
        (df["gain_subsize_marginal_thresh"] > 0) &
        (df["loss_subsize_marginal_thresh"] > 0)
    )
    df_filt = df[mask].copy()
    print(f"  Genes after filtering (rate > 0): {len(df_filt)} / {len(df)}")

    # ── Reference subsize (used for scaling on new trees) ────────────────────
    ecoli_mean_subsize = float((df_filt["gain_subsize_marginal_thresh"].mean() + df_filt["loss_subsize_marginal_thresh"].mean())/2)
    print(f"  Mean gain_subsize_marginal_thresh (scaling ref): {ecoli_mean_subsize:.6f}")

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
        "kde":               kde,
        "ecoli_total_bl":    ecoli_total_bl,
        "ecoli_mean_subsize": ecoli_mean_subsize,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"KDE saved to {args.out}")


if __name__ == "__main__":
    main()
