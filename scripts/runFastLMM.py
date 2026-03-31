#!/usr/bin/env python
"""
Run PySEER (FastLMM) in native GWAS mode: one phenotype vs all genotypes.

Inputs:
  --trait_file  : Reformatted binary trait CSV (samples × traits)
  --working_dir : Scratch directory for temp files
  --kinship     : Phylogenetic similarity matrix (from phylogeny_distance.py)
  --pair_labels : CSV with columns trait1, trait2, direction (from generateData)
  --outfile     : Output CSV path

For each of the 1200 traits as phenotype, pyseer is run with all other traits
as genotypes.  Results are aggregated and evaluated against pair_labels.

Output schema:
  phenotype, genotype, beta, lrt-pvalue, label
  (label = direction from pair_labels for the (phenotype, genotype) pair,
   or NaN if the pair is not in pair_labels)
"""

import argparse
import os
import subprocess
import numpy as np
import pandas as pd
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

parser = argparse.ArgumentParser(description="Run PySEER FastLMM in GWAS mode.")
parser.add_argument("--trait_file",  required=True, help="Path to the trait annotation file.")
parser.add_argument("--working_dir", required=True, help="Directory to store scratch files.")
parser.add_argument("--kinship",     required=True, help="Path to the kinship similarity matrix.")
parser.add_argument("--pair_labels", required=True, help="Path to pair_labels.csv.")
parser.add_argument("--outfile",     required=True, help="Final output filename.")
args = parser.parse_args()

os.makedirs(args.working_dir, exist_ok=True)
os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

df           = pd.read_csv(args.trait_file, index_col=0)
pair_labels  = pd.read_csv(args.pair_labels)
kinship_file = args.kinship

# Build pair-label lookup: frozenset({t1, t2}) → direction
pair_label_lookup = {
    frozenset({str(r["trait1"]), str(r["trait2"])}): int(r["direction"])
    for _, r in pair_labels.iterrows()
}

# Build per-phenotype genotype lists from pair_labels only
pheno_to_genos: dict[str, list[str]] = {}
for _, r in pair_labels.iterrows():
    t1, t2 = str(r["trait1"]), str(r["trait2"])
    pheno_to_genos.setdefault(t1, []).append(t2)
    pheno_to_genos.setdefault(t2, []).append(t1)


def run_pyseer_single(pheno_col: str) -> pd.DataFrame:
    """Run pyseer with pheno_col as phenotype vs its paired genotypes only."""
    geno_cols = pheno_to_genos.get(pheno_col, [])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as pheno_f, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as pres_f:

        pheno_path = pheno_f.name
        pres_path  = pres_f.name

        # Phenotype file: sample<TAB>value (no header)
        df[[pheno_col]].to_csv(pheno_path, sep='\t', header=False)

        # Presence/absence file: rows = variants (traits), cols = samples
        geno_df = df[geno_cols].T.reset_index()
        geno_df.rename(columns={"index": "Gene"}, inplace=True)
        geno_df.to_csv(pres_path, sep='\t', index=False)

    try:
        pyseer_cmd = [
            'pyseer',
            '--phenotypes', pheno_path,
            '--pres',       pres_path,
            '--similarity', kinship_file,
            '--lmm',
        ]
        result = subprocess.run(pyseer_cmd, text=True, capture_output=True, check=True)
        out_df = pd.read_csv(StringIO(result.stdout), sep='\t')
        out_df["phenotype"] = pheno_col
        if "variant" in out_df.columns:
            out_df = out_df.rename(columns={"variant": "genotype"})
        return out_df[["phenotype", "genotype", "beta", "lrt-pvalue"]]
    except subprocess.CalledProcessError as e:
        print(f"  pyseer failed for phenotype {pheno_col}: {e.stderr[:200]}", flush=True)
        return pd.DataFrame()
    finally:
        os.unlink(pheno_path)
        os.unlink(pres_path)


pheno_list = list(pheno_to_genos.keys())
print(f"Running pyseer for {len(pheno_list)} phenotypes (pair_labels only)...", flush=True)

results = []
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(run_pyseer_single, col): col for col in pheno_list}
    for i, future in enumerate(as_completed(futures), 1):
        col = futures[future]
        try:
            res = future.result()
            if not res.empty:
                results.append(res)
        except Exception as e:
            print(f"  Error for phenotype {col}: {e}", flush=True)
        if i % 100 == 0:
            print(f"  Completed {i}/{len(pheno_list)} phenotypes", flush=True)

if not results:
    print("WARNING: no pyseer results collected.", flush=True)
    pd.DataFrame(columns=["phenotype", "genotype", "beta", "lrt-pvalue", "label"]).to_csv(
        args.outfile, index=False)
else:
    agg = pd.concat(results, ignore_index=True)

    def get_label(row):
        key = frozenset({str(row["phenotype"]), str(row["genotype"])})
        return pair_label_lookup.get(key, np.nan)

    agg["label"] = agg.apply(get_label, axis=1)
    agg.to_csv(args.outfile, index=False)
    print(f"Saved FastLMM results to {args.outfile} ({len(agg)} rows)", flush=True)
