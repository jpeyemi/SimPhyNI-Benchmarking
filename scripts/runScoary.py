#!/usr/bin/env python
"""
Run Scoary2 on labeled trait pairs.

Inputs (CLI):
  --systems     : Reformatted binary trait CSV (samples × traits)
  --tree        : Reformatted Newick tree file
  --pair_labels : CSV with columns trait1, trait2, direction
  --outfile     : Output CSV path

For each labeled pair (trait1, trait2) in pair_labels:
  - trait2 acts as the "genotype" (genes file, transposed)
  - trait1 acts as the "phenotype"
  - scoary2 evaluates whether the gene is associated with the phenotype

Output: aggregated scoary2 results with ground-truth label column.
"""

import argparse
import os
import subprocess
import tempfile
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description="Run Scoary2 on labeled trait pairs.")
parser.add_argument("--systems",     required=True)
parser.add_argument("--tree",        required=True)
parser.add_argument("--pair_labels", required=True)
parser.add_argument("--outfile",     required=True)
args = parser.parse_args()

outdir = os.path.dirname(args.outfile)
os.makedirs(outdir, exist_ok=True)

data        = pd.read_csv(args.systems, index_col=0)
pair_labels = pd.read_csv(args.pair_labels)
tree        = args.tree


def run_scoary_pair(trait1, trait2, label):
    with tempfile.TemporaryDirectory() as tmpdir:
        scoary_outdir = os.path.join(tmpdir, "scoary_output")

        t1_path = os.path.join(tmpdir, "t1.csv")
        t2_path = os.path.join(tmpdir, "t2.csv")

        # Phenotype: trait1
        data[[trait1]].to_csv(t1_path)
        # Genes: trait2 transposed (one gene per row)
        data[[trait2]].transpose().to_csv(t2_path)

        scoary_cmd = (
            f"scoary2 --genes {t2_path} "
            f"--gene-data-type gene-count:, "
            f"--traits {t1_path} "
            f"--outdir {scoary_outdir} "
            f"--n-permut 1000 "
            f"--newicktree={tree}"
        )
        subprocess.run(scoary_cmd, shell=True, executable="/bin/bash",
                       capture_output=True)

        result_path = glob.glob(f'{scoary_outdir}/traits/*/result.tsv')
        if not result_path:
            return None

        res_df = pd.read_csv(result_path[0], sep='\t')
        res_df['trait1'] = trait1
        res_df['trait2'] = trait2
        res_df['label']  = label
        return res_df


results = Parallel(n_jobs=-1)(
    delayed(run_scoary_pair)(
        str(row["trait1"]), str(row["trait2"]), int(row["direction"])
    )
    for _, row in pair_labels.iterrows()
)
results = [r for r in results if r is not None]

if results:
    agg = pd.concat(results, ignore_index=True)
    agg.to_csv(args.outfile, index=False)
    print(f"Saved Scoary results to {args.outfile} ({len(agg)} rows)")
else:
    print("WARNING: no Scoary results collected.")
    pd.DataFrame(columns=["trait1", "trait2", "label"]).to_csv(args.outfile, index=False)
