#!/usr/bin/env python
"""Convert binary traits CSV to coinfinder long edge format (trait<TAB>sample per present cell)."""
import argparse, pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--traits', required=True)
p.add_argument('--edges',  required=True)
args = p.parse_args()

df = pd.read_csv(args.traits, index_col=0)
present = df.stack()
present = present[present == 1]
with open(args.edges, 'w') as f:
    for (sample, trait) in present.index:
        f.write(f"{trait}\t{sample}\n")
