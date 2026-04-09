#!/usr/bin/env python
"""
Run SimPhyNI on synthetic data using FLOW_ORIGINAL_DIST ACR parameters.

Inputs:
  --pastml      : ACR output CSV from run_ancestral_reconstruction.py (--uncertainty both)
  --systems     : Synthetic trait CSV (samples × traits)
  --tree        : Reformatted Newick tree
  --pair_labels : CSV with columns trait1, trait2, direction (from generateData)
  --outfile     : Output CSV path for simphyni results

Pairs are read from pair_labels.csv rather than inferred from column order.
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
parser.add_argument("--pastml",      required=True, help="Path to ACR output CSV")
parser.add_argument("--systems",     required=True, help="Path to synthetic traits CSV")
parser.add_argument("--tree",        required=True, help="Path to reformatted Newick tree")
parser.add_argument("--pair_labels", required=True, help="Path to pair_labels.csv")
parser.add_argument("--outfile",     required=True, help="Output CSV path for results")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

# ----------------------
# Load and build params
# ----------------------
acr_df     = pd.read_csv(args.pastml)
sim_params = build_sim_params(acr_df, counting='FLOW', subsize='ORIGINAL', no_threshold=False)

pair_labels = pd.read_csv(args.pair_labels)

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
    run_traits=1,
)

# ----------------------
# Pair setup from pair_labels
# ----------------------
cols  = list(sim.obsdf_modified.columns)
col_set = set(cols)

pairs = [
    (str(row["trait1"]), str(row["trait2"]))
    for _, row in pair_labels.iterrows()
    if str(row["trait1"]) in col_set and str(row["trait2"]) in col_set
]

sim.pairs, sim.obspairs = sim._get_pair_data2(sim.obsdf_modified, pairs)
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

# Build label lookup from pair_labels
label_lookup = {}
for _, row in pair_labels.iterrows():
    t1, t2 = str(row["trait1"]), str(row["trait2"])
    label_lookup[frozenset({t1, t2})] = int(row["direction"])

res = res.copy()
res["label"] = res.apply(
    lambda r: label_lookup.get(frozenset({str(r["T1"]), str(r["T2"])}), None),
    axis=1
)

res = res.rename(columns={"pval_naive": "p-value"})
res.to_csv(args.outfile, index=False)
print(f"Saved SimPhyNI results to {args.outfile}")
