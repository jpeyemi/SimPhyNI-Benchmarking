"""
runParamTraversal_v2.py — Updated parameter traversal benchmarking script.

Fixes over v1 (runParamTraversal.py):
  - Each (method, statistic) grid cell uses the SAME statistic for both obspairs
    and the simulated null distribution (v1 hardcoded LOR for the null).
  - Simulation runs ONCE per (counting × method) and is compiled 7 times — one
    per pair statistic — instead of being re-run per statistic.
  - KDE p-values only for LOR and log_add_ratio (continuous, unbounded).
    All other statistics use empirical counting against their own null.
  - FLOW counting method supported when pastml CSV contains gains_flow column.
  - Output has a Counting column; all other columns match v1.

Usage:
    python scripts/runParamTraversal_v2.py \\
        --pastml  1-PastML-api/{tree}/es{es}/pastmlout.csv \\
        --systems 0-formatting/{tree}/es{es}/reformated_systems.csv \\
        --tree    0-formatting/{tree}/reformated_tree.nwk \\
        --pair_labels 0-formatting/{tree}/es{es}/pair_labels.csv \\
        --outfile 2-Results/{tree}/es{es}/paramtraversal_v2.csv
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as sm
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    auc,
    roc_auc_score,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "SimPhyNI"))
sys.path.insert(0, os.path.dirname(__file__))

from simphyni.Simulation.tree_simulator import TreeSimulator
from simphyni.Simulation.simulation import build_sim_params
from simphyni.Simulation.pair_statistics import pair_statistics as PairStatistics

from sim_wrappers import (
    get_lineages_simulate,
    get_lineages_nodist,
    get_lineages_ctmp,
    get_lineages_norm,
    compile_results,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run updated parameter traversal benchmark.")
parser.add_argument("--pastml",      required=True, help="Path to ACR output CSV")
parser.add_argument("--systems",     required=True, help="Path to synthetic traits CSV")
parser.add_argument("--tree",        required=True, help="Path to reformatted Newick tree")
parser.add_argument("--pair_labels", required=True, help="Path to pair_labels.csv")
parser.add_argument("--outfile",     required=True, help="Output CSV path for results")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

# Statistics that use KDE for p-value estimation (continuous, unbounded support).
# All others use empirical counting against the simulated null.
KDE_STATS = {"lor", "log_add_ratio"}

PAIR_STATISTICS = [
    ("lor",           PairStatistics._log_odds_ratio_statistic),
    ("log_add_ratio", PairStatistics._log_add_ratio_statistic),
    ("treewas",       PairStatistics._treewas_statistic),
    ("jaccard",       PairStatistics._jaccard_index_statistic),
    ("mi",            PairStatistics._mutual_information_statistic),
    ("zscore",        PairStatistics.z_statistic),
    ("count",         PairStatistics.count_statistic),
]

SIMULATION_METHODS = [
    ("simulate",        get_lineages_simulate),
    ("simulate_nodist", get_lineages_nodist),
    ("simulate_ctmp",   get_lineages_ctmp),
    ("simulate_norm",   get_lineages_norm),
]

thresholds        = [0.05, 0.01, 0.001]
correction_conds  = [("fdr_bh", 0.01), ("fdr_by", 0.01), ("bonferroni", 0.01)]
significance_conds = thresholds + correction_conds

# Map correction name to the pre-computed corrected p-value column
CORRECTION_COL = {
    "fdr_bh":     "pval_bh",
    "fdr_by":     "pval_by",
    "bonferroni": "pval_bonf",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

acr_df = pd.read_csv(args.pastml)
HAS_FLOW = "gains_flow" in acr_df.columns
counting_methods = ["JOINT", "FLOW"] if HAS_FLOW else ["JOINT"]

print(f"Counting methods detected: {counting_methods}")

# Use TreeSimulator only for data preparation (tree, obsdf_modified, pair filtering).
# Its internal pair_statistic is hardcoded to LOR — we override that below.
# Pass acr_df as a DataFrame to TreeSimulator so it skips its internal
# pd.read_csv(index_col=0), which would make `gene` the index and break the
# `"gene" in self.pastml` column check.
Sim = TreeSimulator(tree=args.tree, pastmlfile=acr_df, obsdatafile=args.systems)
Sim.initialize_simulation_parameters(collapse_threshold=0.00, pre_filter=False)

gene_name_set = set(Sim.obsdf_modified.columns)
pair_labels_df = pd.read_csv(args.pair_labels)

# Filter labeled pairs to those present in obsdf_modified
filtered = [
    (str(r["trait1"]), str(r["trait2"]), int(r["direction"]))
    for _, r in pair_labels_df.iterrows()
    if str(r["trait1"]) in gene_name_set and str(r["trait2"]) in gene_name_set
]

if len(filtered) == 0:
    raise ValueError(
        "No labeled pairs overlap with obsdf_modified.columns. "
        "Check that trait1/trait2 column names match gene names in the ACR CSV."
    )

valid_gene_pairs = np.array([(t1, t2) for t1, t2, _ in filtered])
labels = np.array([d for _, _, d in filtered])

print(f"Labeled pairs: {len(valid_gene_pairs)} (after filtering to observed genes)")

# ---------------------------------------------------------------------------
# Pre-compute observed statistic values from obsdf_modified
# ---------------------------------------------------------------------------

obs_np   = Sim.obsdf_modified.to_numpy().astype(bool)
col_to_i = {c: i for i, c in enumerate(Sim.obsdf_modified.columns)}
pair_i   = np.array([(col_to_i[a], col_to_i[b]) for a, b in valid_gene_pairs])


def obs_stat(stat_fn):
    """Compute observed statistic values for all valid_gene_pairs."""
    return stat_fn(obs_np[:, pair_i[:, 0]], obs_np[:, pair_i[:, 1]])


# ---------------------------------------------------------------------------
# Build trait_params per counting method (once each)
# ---------------------------------------------------------------------------

trait_params_by_counting = {
    cm: build_sim_params(acr_df, counting=cm, subsize="ORIGINAL")
    for cm in counting_methods
}

# Filter valid_gene_pairs to genes present in trait_params
# (trait_params and obsdf_modified should share the same gene set, but guard
#  in case ACR coverage differs from observation data coverage)
tp_gene_set = set(trait_params_by_counting[counting_methods[0]].index)
valid_mask = np.array(
    [t1 in tp_gene_set and t2 in tp_gene_set for t1, t2 in valid_gene_pairs]
)
if not valid_mask.all():
    n_dropped = int((~valid_mask).sum())
    print(f"Warning: dropping {n_dropped} pairs not in trait_params gene set")
    valid_gene_pairs = valid_gene_pairs[valid_mask]
    labels = labels[valid_mask]
    pair_i = pair_i[valid_mask]

# ---------------------------------------------------------------------------
# Metric helpers (carried verbatim from runParamTraversal.py)
# ---------------------------------------------------------------------------

def calcFDR(labels, pvalues, directions, threshold=0.05):
    labels     = np.array(labels)
    pvalues    = np.array(pvalues)
    directions = np.array(directions)
    significant = pvalues <= threshold

    predicted_positives = (directions == 1) & significant
    true_positives  = np.sum((labels == 1) & predicted_positives)
    false_positives = np.sum((labels != 1) & predicted_positives)
    fdr_positive = (false_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0 else 0.0)

    predicted_negatives = (directions == -1) & significant
    true_negatives  = np.sum((labels == -1) & predicted_negatives)
    false_negatives = np.sum((labels != -1) & predicted_negatives)
    fdr_negative = (false_negatives / (true_negatives + false_negatives)
                    if (true_negatives + false_negatives) > 0 else 0.0)

    return fdr_positive, fdr_negative


def calcFPR(labels, pvalues, directions, threshold=0.05):
    labels     = np.array(labels)
    pvalues    = np.array(pvalues)
    directions = np.array(directions)
    significant = pvalues <= threshold

    predicted_positives = (directions == 1) & significant
    false_positives_pos  = np.sum((labels != 1) & predicted_positives)
    total_negatives_pos  = np.sum(labels != 1)
    fpr_positive = (false_positives_pos / total_negatives_pos
                    if total_negatives_pos > 0 else 0.0)

    predicted_negatives = (directions == -1) & significant
    false_positives_neg  = np.sum((labels != -1) & predicted_negatives)
    total_negatives_neg  = np.sum(labels != -1)
    fpr_negative = (false_positives_neg / total_negatives_neg
                    if total_negatives_neg > 0 else 0.0)

    return fpr_positive, fpr_negative


def AUCs(labels, pvalues, predicted_directions, dir, name):
    binary_labels = (labels == dir).astype(int)
    pv = pvalues.copy()
    pv[predicted_directions == (dir * -1)] = 1 - pv[predicted_directions == (dir * -1)]
    min_pvalue = 1e-300
    capped_pvalues = np.maximum(pv, min_pvalue)
    try:
        auc_score = roc_auc_score(binary_labels, -np.log10(capped_pvalues))
    except ValueError:
        auc_score = float("nan")
    precision, recall, _ = precision_recall_curve(binary_labels, -np.log10(capped_pvalues))
    pr_auc = auc(recall, precision)
    return auc_score, pr_auc


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate_v2(raw, labels, method_name, stat_name, counting, significance_conds):
    """
    Evaluate one (method, statistic, counting) cell across all significance
    conditions.

    AUC metrics are computed once from the raw p-values (condition-independent)
    and reused across all significance threshold rows.

    Returns a list of row dicts (one per significance condition).
    """
    pvals      = np.array(raw["p-value"])
    directions = np.array(raw["direction"])
    labels_arr = np.array(labels)

    # AUC computed once per (method, stat, counting) from raw p-values
    neg_auc, neg_pr = AUCs(labels_arr, pvals, directions, -1,
                            f"{method_name}_{stat_name}_neg")
    pos_auc, pos_pr = AUCs(labels_arr, pvals, directions,  1,
                            f"{method_name}_{stat_name}_pos")

    rows = []
    for condition in significance_conds:
        if isinstance(condition, tuple):
            correction, significance_threshold = condition
            cond_pvals = np.array(raw[CORRECTION_COL[correction]])
        else:
            significance_threshold = condition
            correction = False
            cond_pvals = pvals

        filtered_predictions = np.where(
            cond_pvals < significance_threshold, directions, 0
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, filtered_predictions,
            average=None, labels=[-1, 1], zero_division=0,
        )
        accuracy = float(np.mean(filtered_predictions == labels_arr))

        fdrp, fdrn = calcFDR(labels_arr, cond_pvals, directions, significance_threshold)
        fprp, fprn = calcFPR(labels_arr, cond_pvals, directions, significance_threshold)

        rows.append({
            "Method":             method_name,
            "Statistic":          stat_name,
            "Counting":           counting,
            "Threshold":          significance_threshold,
            "Bonferroni":         correction,
            "Precision_Negative": precision[0],
            "Recall_Negative":    recall[0],
            "F1_Negative":        f1[0],
            "Precision_Positive": precision[1],
            "Recall_Positive":    recall[1],
            "F1_Positive":        f1[1],
            "Accuracy":           accuracy,
            "AUC_ROC_Negative":   neg_auc,
            "PR_AUC_Negative":    neg_pr,
            "AUC_ROC_Positive":   pos_auc,
            "PR_AUC_Positive":    pos_pr,
            "FDR_Negative":       fdrn,
            "FPR_Negative":       fprn,
            "FDR_Positive":       fdrp,
            "FPR_Positive":       fprp,
        })
    return rows


# ---------------------------------------------------------------------------
# Main loop — simulate once per (counting × method), compile per statistic
# ---------------------------------------------------------------------------

results_rows = []

for counting in counting_methods:
    tp = trait_params_by_counting[counting]

    for method_name, get_lin_fn in SIMULATION_METHODS:
        print(f"\n--- Simulating: counting={counting}, method={method_name} ---")

        if method_name == "simulate_nodist":
            lineages, mappingr, to_idx = get_lin_fn(
                Sim.tree, acr_df, counting=counting, subsize="ORIGINAL"
            )
        else:
            lineages, mappingr, to_idx = get_lin_fn(Sim.tree, tp)

        for stat_name, stat_fn in PAIR_STATISTICS:
            print(f"  Compiling: stat={stat_name}", flush=True)
            use_kde  = stat_name in KDE_STATS
            obspairs = obs_stat(stat_fn)

            raw = compile_results(
                lineages, valid_gene_pairs, obspairs,
                stat_fn, use_kde, mappingr, to_idx,
            )

            # Multiple testing correction (compute once, store as columns)
            pvals_arr = np.array(raw["p-value"])
            raw["pval_bh"]   = sm.multipletests(pvals_arr, alpha=0.01, method="fdr_bh")[1]
            raw["pval_by"]   = sm.multipletests(pvals_arr, alpha=0.01, method="fdr_by")[1]
            raw["pval_bonf"] = sm.multipletests(pvals_arr, alpha=0.01, method="bonferroni")[1]

            rows = evaluate_v2(raw, labels, method_name, stat_name, counting,
                               significance_conds)
            results_rows.extend(rows)

pd.DataFrame(results_rows).to_csv(args.outfile, index=False)
print(f"\nResults saved to {args.outfile}  ({len(results_rows)} rows)")
