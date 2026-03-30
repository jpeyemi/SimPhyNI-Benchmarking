#!/usr/bin/env python
"""
Run SimPhyNI on synthetic data using FLOW_ORIGINAL_DIST ACR parameters.

Inputs:
  --pastml   : ACR output CSV from run_ancestral_reconstruction.py (--uncertainty both)
  --systems  : Synthetic trait CSV (samples × traits)
  --tree     : Reformatted Newick tree
  --outfile  : Output CSV path for simphyni results

Pair setup: consecutive pairs (0,1), (2,3), ... matching synthetic data structure.
Classification labels: [0]*3000 + [-1]*300 + [1]*300
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd

from simphyni.Simulation.simulation import build_sim_params
from simphyni.Simulation.tree_simulator import TreeSimulator

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser(description="Run SimPhyNI with FLOW_ORIGINAL_DIST.")
parser.add_argument("--pastml",   required=True, help="Path to ACR output CSV")
parser.add_argument("--systems",  required=True, help="Path to synthetic traits CSV")
parser.add_argument("--tree",     required=True, help="Path to reformatted Newick tree")
parser.add_argument("--outfile",  required=True, help="Output CSV path for results")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

# ----------------------
# Load and build params
# ----------------------
acr_df = pd.read_csv(args.pastml)
sim_params = build_sim_params(acr_df, counting='FLOW', subsize='ORIGINAL', no_threshold=False)

# ----------------------
# Build TreeSimulator
# ----------------------
sim = TreeSimulator(
    tree=args.tree,
    pastmlfile=sim_params,
    obsdatafile=args.systems,
)

sim.initialize_simulation_parameters(
    collapse_threshold=0.000,
    prevalence_threshold=0.00,
    pre_filter=False,
    run_traits = 1,
)

# ----------------------
# Consecutive-pair setup
# ----------------------
cols = list(sim.obsdf_modified.columns)
refs = list(range(0, len(cols), 2))
pairs = [(cols[i], cols[i + 1]) for i in refs]
sim.pairs, sim.obspairs = sim._get_pair_data2(sim.obsdf_modified, pairs)

# Set total_tests so multiple-test correction in _multiple_test_correction works
sim.total_tests = len(sim.pairs)

# ----------------------
# Run simulation
# ----------------------
print("Running SimPhyNI simulation...")
sim.run_simulation()

# ----------------------
# Collect results + labels
# ----------------------
res = sim.get_results()
# Sort output by sample name to ensure labels align
res = res.sort_values(
    by='T1',
    key=lambda x: x.str.extract(r'(\d+)$')[0].astype(int)
)

# Attach ground-truth labels: first 3000 pairs are neutral, next 300 negative, last 300 positive
labels = [0] * 3000 + [-1] * 300 + [1] * 300
if len(labels) != len(res):
    raise ValueError(
        f"Label length mismatch: expected {len(labels)}, got {len(res)} pairs. "
        "Check synthetic data structure (should be 3600 total pairs)."
    )
res = res.copy()
res['label'] = labels

# Rename pval_naive to p-value for consistency with downstream scripts
res = res.rename(columns={'pval_naive': 'p-value'})

res.to_csv(args.outfile, index=False)
print(f"Saved SimPhyNI results to {args.outfile}")
