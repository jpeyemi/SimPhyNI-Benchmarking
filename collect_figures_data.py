"""
collect_figures_data.py

Collects all outputs from the Snakemake benchmarking and downstream analysis
pipelines into a single flat figure_data/ directory.  The resulting directory
is self-contained and can be downloaded locally so that all figure notebooks
in local_method_comparison/ can run without cluster access.

Source paths
============
Raw method results (per tree × per effect size):
  2-Results/{tree}/{es}/simphyni_results_flow.csv
  2-Results/{tree}/{es}/coinfinder_results.csv
  2-Results/{tree}/{es}/spydrpick_results.csv
  2-Results/{tree}/{es}/fastlmm_results.csv
  2-Results/{tree}/{es}/scoary_results.csv
  2-Results/{tree}/{es}/pagel_results.csv        (or pagel_chunks/*.csv)
  2-Results/{tree}/{es}/htreewas_terminal.csv

Ground-truth + D-statistic labels:
  0-formatting/{tree}/{es}/pair_labels.csv

Pre-computed analysis outputs:
  analysis/pr_auc_bins.csv
  analysis/all_metrics.csv
  analysis/pr_auc_summary.csv

ACR stability benchmarks:
  benchmark-acr/bench_{0,1,2}/acr_benchmark/stability_trajectory.csv
  benchmark-acr/bench_{0,1,2}/acr_benchmark/stability.csv
  benchmark-acr/bench_{0,1,2}/acr_benchmark/method_ranking.csv

Output files
============
figure_data/per_pair_results.csv   — per-pair detection outcomes for all methods
figure_data/fpr_bins.csv           — FPR binned by D-statistic per method
figure_data/simphyni_effects.csv   — SimPhyNI inferred effect sizes per pair
figure_data/pr_auc_bins.csv        — (copy) PR-AUC binned by D-statistic
figure_data/all_metrics.csv        — (copy) aggregate metrics per method/tree/es
figure_data/pr_auc_summary.csv     — (copy) PR-AUC summary by method/stratum
figure_data/paramtraversal.csv       — concatenated paramtraversal grids (tree × es)
figure_data/stability_trajectory.csv — concatenated ACR stability trajectories
figure_data/stability.csv          — concatenated ACR calibration metrics
figure_data/method_ranking.csv     — concatenated ACR method rankings

Usage (on cluster):
    conda run -n simphyni_dev python collect_figures_data.py
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / '2-Results'
FORMAT_ROOT  = REPO_ROOT / '0-formatting'
ANALYSIS_DIR = REPO_ROOT / 'analysis'
BENCH_ROOT   = REPO_ROOT / 'benchmark-acr'
OUT_DIR      = REPO_ROOT / 'figure_data'

EFFECT_SIZES = ['es0', 'es1', 'es2', 'es3', 'es4', 'es5']
ES_INPUT_MAP = {'es0': 3.0, 'es1': 2.0, 'es2': 1.0, 'es3': 0.75, 'es4': 0.5, 'es5': 0.25}

# ── Bootstrap analysis imports ────────────────────────────────────────────────
# Reuse the per-pair loader infrastructure from the analysis scripts.
sys.path.insert(0, str(ANALYSIS_DIR))
from logistic_regression_d_vs_performance import collect_all_pairs  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        warnings.warn(f'Source not found, skipping copy: {src}')
        return
    shutil.copy2(src, dst)
    print(f'  copied  {src.relative_to(REPO_ROOT)}  →  {dst.relative_to(REPO_ROOT)}')


# ── Task 1: per-pair results ──────────────────────────────────────────────────

def collect_per_pair() -> pd.DataFrame:
    """
    Call collect_all_pairs() from the logistic regression script.
    Returns long-format DataFrame:
      method, tree, effect_size, d_statistic, d_stratum, direction,
      raw_pvalue, corr_pvalue, detected, pred_direction
    """
    print('\n[1] Collecting per-pair detection outcomes …')
    master = collect_all_pairs()
    print(f'    → {len(master):,} rows, {master["method"].nunique()} methods')
    return master


# ── Task 2: FPR bins ──────────────────────────────────────────────────────────

def build_fpr_bins(master: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """
    For each (method, tree, effect_size) group, bin d_statistic into n_bins
    quantile bins and compute FPR (false positive rate on null pairs) within
    each bin.

    Null pairs: direction == 0.
    FPR = detected.mean() within the null-pair subset of each bin.

    Returns DataFrame with columns:
      method, tree, effect_size, d_bin_mid, fpr, n_total, n_neg,
      es_input, d_mid_c, es_input_c
    """
    from logistic_regression_d_vs_performance import ES_INPUT_MAP  # noqa

    print('\n[2] Building FPR bins …')
    rows = []
    for (method, tree, es), grp in master.groupby(
            ['method', 'tree', 'effect_size'], observed=True):
        grp = grp.dropna(subset=['d_statistic', 'raw_pvalue'])
        if len(grp) < 50:
            continue
        try:
            grp = grp.copy()
            grp['d_bin'] = pd.qcut(grp['d_statistic'], q=n_bins,
                                   duplicates='drop')
        except ValueError:
            grp['d_bin'] = pd.cut(grp['d_statistic'], bins=n_bins)

        es_input = ES_INPUT_MAP[es]
        null_grp = grp[grp['direction'] == 0]
        for d_bin, bgrp in null_grp.groupby('d_bin', observed=True):
            rows.append({
                'method':      method,
                'tree':        tree,
                'effect_size': es,
                'd_bin_mid':   bgrp['d_statistic'].mean(),
                'fpr':         bgrp['detected'].mean(),
                'n_total':     len(bgrp),
                'n_neg':       int((bgrp['direction'] == 0).sum()),
                'es_input':    es_input,
            })

    df = pd.DataFrame(rows)
    d_mean  = df['d_bin_mid'].mean()
    es_mean = df['es_input'].mean()
    df['d_mid_c']    = df['d_bin_mid'] - d_mean
    df['es_input_c'] = df['es_input']  - es_mean
    print(f'    → {len(df):,} rows')
    return df


# ── Task 3: SimPhyNI inferred effect sizes ────────────────────────────────────

def collect_simphyni_effects() -> pd.DataFrame:
    """
    Collect SimPhyNI inferred effect sizes (per pair) across all trees and
    effect sizes.  Used by effect_size_vs_interaction_strength_4smm.ipynb.

    Columns: tree, effect_size, es_input, trait1, trait2, label, direction,
             inferred_effect_size, raw_pvalue
    """
    from logistic_regression_d_vs_performance import ES_INPUT_MAP  # noqa

    print('\n[3] Collecting SimPhyNI inferred effect sizes …')
    records = []
    trees = sorted(d.name for d in RESULTS_ROOT.iterdir() if d.is_dir())
    for tree in trees:
        for es in EFFECT_SIZES:
            path = RESULTS_ROOT / tree / es / 'simphyni_results_flow.csv'
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df.rename(columns={'T1': 'trait1', 'T2': 'trait2',
                                    'p-value': 'raw_pvalue',
                                    'effect size': 'inferred_effect_size'})
            df['tree'] = tree
            df['effect_size'] = es
            df['es_input'] = ES_INPUT_MAP[es]
            records.append(df[['tree', 'effect_size', 'es_input',
                                'trait1', 'trait2', 'label', 'direction',
                                'inferred_effect_size', 'raw_pvalue']])

    result = pd.concat(records, ignore_index=True)
    print(f'    → {len(result):,} rows')
    return result


# ── Task 4: paramtraversal grid ──────────────────────────────────────────────

def collect_paramtraversal() -> pd.DataFrame:
    """
    Concatenate paramtraversal.csv files from 2-Results/{tree}/{es}/ across
    all available trees and effect sizes.  Adds 'tree', 'effect_size', and
    'es_input' columns so the notebook can group/filter without re-reading
    directory names.

    Columns: tree, effect_size, es_input, Method, Statistic, Threshold,
             Bonferroni, Precision_Negative, Recall_Negative, F1_Negative,
             Precision_Positive, Recall_Positive, F1_Positive,
             AUC_ROC_Negative, PR_AUC_Negative, AUC_ROC_Positive,
             PR_AUC_Positive, FDR_Negative, FPR_Negative, FDR_Positive,
             FPR_Positive
    (Accuracy is dropped — it conflates pos/neg performance.)
    """
    print('\n[4] Collecting paramtraversal grid results …')
    frames = []
    trees = sorted(d.name for d in RESULTS_ROOT.iterdir() if d.is_dir())
    for tree in trees:
        for es in EFFECT_SIZES:
            path = RESULTS_ROOT / tree / es / 'paramtraversal.csv'
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df.drop(columns=['Accuracy'], errors='ignore', inplace=True)
            df.insert(0, 'es_input', ES_INPUT_MAP[es])
            df.insert(0, 'effect_size', es)
            df.insert(0, 'tree', tree)
            frames.append(df)
            print(f'    read  {path.relative_to(REPO_ROOT)}  ({len(df):,} rows)')

    if not frames:
        warnings.warn(f'No paramtraversal.csv files found under {RESULTS_ROOT}')
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    print(f'    → {len(result):,} rows total')
    return result


# ── Task 5: stability trajectories ───────────────────────────────────────────

def collect_stability_trajectories() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Concatenate stability_trajectory.csv, stability.csv, and
    method_ranking.csv across all benchmark runs (bench_0, bench_1, bench_2).

    Adds a 'bench' column to each table.
    """
    print('\n[5] Collecting ACR stability trajectories …')
    traj_frames, stab_frames, rank_frames = [], [], []

    bench_dirs = sorted(BENCH_ROOT.glob('bench_*'))
    if not bench_dirs:
        warnings.warn(f'No benchmark directories found under {BENCH_ROOT}')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for bench_dir in bench_dirs:
        bench = bench_dir.name
        acr_dir = bench_dir / 'acr_benchmark'

        for fname, frames in [
            ('stability_trajectory.csv', traj_frames),
            ('stability.csv',            stab_frames),
            ('method_ranking.csv',       rank_frames),
        ]:
            fpath = acr_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath)
                df.insert(0, 'bench', bench)
                frames.append(df)
                print(f'    read  {fpath.relative_to(REPO_ROOT)}  ({len(df):,} rows)')
            else:
                warnings.warn(f'Not found: {fpath}')

    traj = pd.concat(traj_frames, ignore_index=True) if traj_frames else pd.DataFrame()
    stab = pd.concat(stab_frames, ignore_index=True) if stab_frames else pd.DataFrame()
    rank = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
    print(f'    → stability_trajectory: {len(traj):,} rows')
    return traj, stab, rank


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f'Output directory: {OUT_DIR}')

    # 1. Per-pair detection outcomes (used by pr-auc and fpr notebooks)
    master = collect_per_pair()
    out = OUT_DIR / 'per_pair_results.csv'
    master.to_csv(out, index=False)
    print(f'  saved  {out.relative_to(REPO_ROOT)}')

    # 2. FPR bins
    fpr_bins = build_fpr_bins(master)
    out = OUT_DIR / 'fpr_bins.csv'
    fpr_bins.to_csv(out, index=False)
    print(f'  saved  {out.relative_to(REPO_ROOT)}')

    # 3. SimPhyNI inferred effect sizes
    effects = collect_simphyni_effects()
    out = OUT_DIR / 'simphyni_effects.csv'
    effects.to_csv(out, index=False)
    print(f'  saved  {out.relative_to(REPO_ROOT)}')

    # 4. Paramtraversal grid
    paramtraversal = collect_paramtraversal()
    if not paramtraversal.empty:
        out = OUT_DIR / 'paramtraversal.csv'
        paramtraversal.to_csv(out, index=False)
        print(f'  saved  {out.relative_to(REPO_ROOT)}')

    # 5. Stability trajectories
    traj, stab, rank = collect_stability_trajectories()
    for df, name in [(traj, 'stability_trajectory.csv'),
                     (stab, 'stability.csv'),
                     (rank, 'method_ranking.csv')]:
        if not df.empty:
            out = OUT_DIR / name
            df.to_csv(out, index=False)
            print(f'  saved  {out.relative_to(REPO_ROOT)}')

    # 6. Copy pre-computed analysis outputs
    print('\n[6] Copying pre-computed analysis outputs …')
    for fname in ['pr_auc_bins.csv', 'all_metrics.csv', 'pr_auc_summary.csv']:
        _copy(ANALYSIS_DIR / fname, OUT_DIR / fname)

    print('\nDone.  figure_data/ contains:')
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f'  {f.name:<40} {size_kb:>6} KB')


if __name__ == '__main__':
    main()
