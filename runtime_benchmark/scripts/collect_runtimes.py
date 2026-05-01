#!/usr/bin/env python
"""
Collect Snakemake benchmark TSVs and produce runtime box-and-whisker plots.

Snakemake benchmark columns: s, h:m:s, max_rss, max_vms, max_uss, max_pss,
                              io_in, io_out, mean_load, cpu_time
We use the `s` column (wall-clock seconds).

Outputs
-------
  runtime_boxplot.svg        — combined, one box per method (27 points each)
  runtime_boxplot_facet.svg  — 3×3 facet by (n_leaves, n_traits)

Usage
-----
  python collect_runtimes.py \
      --benchmark_dir benchmarks \
      --out_plot      runtime_boxplot.svg \
      --out_facet     runtime_boxplot_facet.svg
"""

import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Color scheme ─────────────────────────────────────────────────────────────
METHOD_ORDER = ['SimPhyNI', 'FaST-LMM', 'TreeWAS', 'Pagel', 'Scoary', 'Coinfinder']

method_colors = {
    'SimPhyNI':   '#1f77b4',
    'FaST-LMM':   '#ff7f0e',
    'TreeWAS':    '#2ca02c',
    'Pagel':      '#d62728',
    'Scoary':     '#9467bd',
    'Coinfinder': '#8c564b',
}

# Map benchmark TSV filenames to display names
TSV_TO_METHOD = {
    'SimPhyNI.tsv':   'SimPhyNI',
    'FaST-LMM.tsv':   'FaST-LMM',
    'TreeWAS.tsv':    'TreeWAS',
    'Pagel.tsv':      'Pagel',
    'Scoary.tsv':     'Scoary',
    'Coinfinder.tsv': 'Coinfinder',
}


def parse_dataset(dataset_dir):
    """Parse 'nl_nt_repR' → (n_leaves, n_traits, rep)."""
    m = re.fullmatch(r'(\d+)_(\d+)_rep(\d+)', dataset_dir)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_benchmarks(benchmark_dir):
    records = []
    for dataset_dir in os.listdir(benchmark_dir):
        parts = parse_dataset(dataset_dir)
        if parts is None:
            continue
        n_leaves, n_traits, rep = parts
        dset_path = os.path.join(benchmark_dir, dataset_dir)
        if not os.path.isdir(dset_path):
            continue
        for tsv_file, method in TSV_TO_METHOD.items():
            tsv_path = os.path.join(dset_path, tsv_file)
            if not os.path.exists(tsv_path):
                continue
            try:
                df = pd.read_csv(tsv_path, sep='\t')
                seconds = float(df['s'].iloc[0])
                records.append({
                    'method':   method,
                    'n_leaves': n_leaves,
                    'n_traits': n_traits,
                    'rep':      rep,
                    'seconds':  seconds,
                })
            except Exception as e:
                print(f"  Warning: could not parse {tsv_path}: {e}")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Combined boxplot
# ─────────────────────────────────────────────────────────────────────────────
def plot_combined(df, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    methods_present = [m for m in METHOD_ORDER if m in df['method'].unique()]
    positions = range(len(methods_present))

    boxes = []
    for pos, method in zip(positions, methods_present):
        data = df.loc[df['method'] == method, 'seconds'].values
        bp = ax.boxplot(
            data,
            positions=[pos],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=method_colors[method], alpha=0.75),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='#333333'),
            capprops=dict(color='#333333'),
            flierprops=dict(marker='o', markersize=4,
                            markerfacecolor=method_colors[method],
                            markeredgecolor='#333333', alpha=0.7),
        )
        boxes.append(bp)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(methods_present, fontsize=12)
    ax.set_ylabel('Wall-clock time (s)', fontsize=12)
    ax.set_title('Runtime comparison across 27 datasets\n(3 tree sizes × 3 trait counts × 3 replicates)',
                 fontsize=13)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f'{x:g}s' if x < 3600 else f'{x/3600:.1f}h'))
    ax.axhline(10800, color='red', linewidth=1, linestyle='--', alpha=0.5,
               label='3h limit')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    fig.tight_layout()
    for ext in ('svg', 'png'):
        fig.savefig(out_path.replace('.svg', f'.{ext}'), dpi=150)
    plt.close(fig)
    print(f"Saved combined plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Faceted boxplot (3 tree sizes × 3 trait counts)
# ─────────────────────────────────────────────────────────────────────────────
def plot_facet(df, out_path):
    n_leaves_vals = sorted(df['n_leaves'].unique())
    n_traits_vals = sorted(df['n_traits'].unique())
    methods_present = [m for m in METHOD_ORDER if m in df['method'].unique()]

    fig, axes = plt.subplots(
        len(n_leaves_vals), len(n_traits_vals),
        figsize=(5 * len(n_traits_vals), 4 * len(n_leaves_vals)),
        sharex=True,
    )
    if len(n_leaves_vals) == 1 and len(n_traits_vals) == 1:
        axes = np.array([[axes]])
    elif len(n_leaves_vals) == 1:
        axes = axes[np.newaxis, :]
    elif len(n_traits_vals) == 1:
        axes = axes[:, np.newaxis]

    for i, nl in enumerate(n_leaves_vals):
        for j, nt in enumerate(n_traits_vals):
            ax = axes[i][j]
            subset = df[(df['n_leaves'] == nl) & (df['n_traits'] == nt)]
            positions = range(len(methods_present))
            for pos, method in zip(positions, methods_present):
                data = subset.loc[subset['method'] == method, 'seconds'].values
                if len(data) == 0:
                    continue
                ax.boxplot(
                    data,
                    positions=[pos],
                    widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor=method_colors[method], alpha=0.75),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='#333333'),
                    capprops=dict(color='#333333'),
                    flierprops=dict(marker='o', markersize=3,
                                    markerfacecolor=method_colors[method],
                                    markeredgecolor='#333333', alpha=0.6),
                )
            ax.set_xticks(list(positions))
            ax.set_xticklabels(methods_present, rotation=30, ha='right', fontsize=8)
            ax.set_yscale('log')
            ax.axhline(10800, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.grid(axis='y', linestyle=':', alpha=0.4)
            if j == 0:
                ax.set_ylabel('Wall-clock (s)', fontsize=9)
            ax.set_title(f'{nl} leaves, {nt} traits', fontsize=9)

    # Legend
    patches = [mpatches.Patch(facecolor=method_colors[m], label=m, alpha=0.75)
               for m in methods_present]
    fig.legend(handles=patches, loc='lower center', ncol=len(methods_present),
               fontsize=9, bbox_to_anchor=(0.5, 0))
    fig.suptitle('Runtime by dataset size', fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ('svg', 'png'):
        fig.savefig(out_path.replace('.svg', f'.{ext}'), dpi=150,
                    bbox_inches='tight')
    plt.close(fig)
    print(f"Saved faceted plot → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark_dir', default='benchmarks')
    p.add_argument('--out_plot',  default='runtime_boxplot.svg')
    p.add_argument('--out_facet', default='runtime_boxplot_facet.svg')
    args = p.parse_args()

    df = load_benchmarks(args.benchmark_dir)
    if df.empty:
        print("No benchmark TSVs found — nothing to plot.")
        return

    print(f"Loaded {len(df)} benchmark entries across "
          f"{df['method'].nunique()} methods and "
          f"{(df[['n_leaves','n_traits','rep']].drop_duplicates().shape[0])} datasets.")

    # Save raw table alongside plots
    csv_path = args.out_plot.replace('.svg', '_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Raw data → {csv_path}")

    plot_combined(df, args.out_plot)
    plot_facet(df, args.out_facet)


if __name__ == '__main__':
    main()
