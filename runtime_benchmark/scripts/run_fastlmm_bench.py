#!/usr/bin/env python
"""
FaST-LMM runtime benchmark — thin pyseer CLI orchestrator.

Runs all-vs-all: for every trait as phenotype, calls pyseer with all other
traits as genotypes.  No pair_labels, no accuracy evaluation — timing only.

Usage:
  python run_fastlmm_bench.py \
      --traits  data/500_100_rep0/traits.csv \
      --kinship results/500_100_rep0/fastlmm/kinship.tsv \
      --outfile results/500_100_rep0/fastlmm/results.csv \
      --threads 16
"""

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--traits',  required=True)
    p.add_argument('--kinship', required=True)
    p.add_argument('--outfile', required=True)
    p.add_argument('--threads', type=int, default=16)
    return p.parse_args()


def run_pyseer_for_phenotype(pheno_col, df, kinship_file):
    """Write temp files then call pyseer CLI; return row count or 0 on failure."""
    geno_cols = [c for c in df.columns if c != pheno_col]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as pf, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as gf:
        pheno_path = pf.name
        geno_path  = gf.name

        # Phenotype: sample<TAB>value (no header)
        df[[pheno_col]].to_csv(pheno_path, sep='\t', header=False)

        # Genotype presence/absence: rows = variants (traits), cols = samples
        geno_df = df[geno_cols].T.reset_index()
        geno_df.rename(columns={"index": "Gene"}, inplace=True)
        geno_df.to_csv(geno_path, sep='\t', index=False)

    try:
        result = subprocess.run(
            ['pyseer', '--lmm',
             '--phenotypes', pheno_path,
             '--pres',       geno_path,
             '--similarity', kinship_file],
            capture_output=True, text=True,
        )
        return len(result.stdout.strip().splitlines()) - 1  # subtract header
    except Exception:
        return 0
    finally:
        os.unlink(pheno_path)
        os.unlink(geno_path)


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    df = pd.read_csv(args.traits, index_col=0)
    traits = list(df.columns)

    print(f"Running pyseer all-vs-all: {len(traits)} traits × {len(df)} samples",
          flush=True)

    total_rows = 0
    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(run_pyseer_for_phenotype, col, df, args.kinship): col
            for col in traits
        }
        for i, fut in enumerate(as_completed(futures), 1):
            total_rows += fut.result() or 0
            if i % max(1, len(traits) // 10) == 0:
                print(f"  {i}/{len(traits)} phenotypes done", flush=True)

    # Write a lightweight sentinel so downstream rules have a concrete output
    with open(args.outfile, 'w') as f:
        f.write(f"total_pyseer_result_rows,{total_rows}\n")
    print(f"Done. Total pyseer result rows: {total_rows}", flush=True)


if __name__ == '__main__':
    main()
