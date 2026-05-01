#!/usr/bin/env python
"""Transpose traits CSV to genes file for scoary2 (traits as rows, samples as columns)."""
import argparse, pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--traits',  required=True)
p.add_argument('--genes_T', required=True)
args = p.parse_args()

pd.read_csv(args.traits, index_col=0).T.to_csv(args.genes_T)
