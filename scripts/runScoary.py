import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import sys
import glob
import re
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
import shutil
from io import StringIO

tree = sys.argv[-2]
obs_data_path = sys.argv[-3]
outfile = sys.argv[-1]
outdir = os.path.dirname(outfile)
os.makedirs(outdir, exist_ok=True)

data = pd.read_csv(obs_data_path, index_col=0)

def run_scoary(i):
    trait1 = data.columns[i]
    trait2 = data.columns[i + 1]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a unique subdirectory within the temporary directory
        scoary_outdir = os.path.join(tmpdir, "scoary_output")
        # os.makedirs(scoary_outdir, exist_ok=False)

        t1_path = os.path.join(tmpdir, "t1.csv")
        t2_path = os.path.join(tmpdir, "t2.csv")

        data[[trait1]].to_csv(t1_path)
        data[[trait2]].transpose().to_csv(t2_path)

        scoary_cmd = f"""
        scoary2 --genes {t2_path} \
                --gene-data-type gene-count:, \
                --traits {t1_path} \
                --outdir {scoary_outdir} \
                --n-permut 1000 \
                --newicktree={tree}
        """

        subprocess.run(scoary_cmd, shell=True, executable="/bin/bash")

        result_path = glob.glob(f'{scoary_outdir}/traits/*/result.tsv')
        if not result_path:
            return None

        res_df = pd.read_csv(result_path[0], sep='\t')
        res_df['trait1'] = trait1
        res_df['trait2'] = trait2
        return res_df


results = Parallel(n_jobs=-1)(delayed(run_scoary)(i) for i in range(0, data.shape[1], 2))
results = [r for r in results if r is not None]

agg = pd.concat(results, ignore_index=True)

agg['num'] = agg['Gene'].str.extract(r'_(\d+)$').astype(int)
agg.sort_values(by='num', inplace=True)
agg['pred'] = agg['odds_ratio'].apply(lambda x: 1 if x > 1 else -1)

labels = np.array([0]*3000 + [-1]*300 + [1]*300)
predictions = agg['pred'].values
pvalues = agg['empirical_p'].fillna(1).values
filtered_preds = np.where(pvalues < 0.05, predictions, 0)

report = classification_report(labels, filtered_preds, target_names=['Negative', 'None', 'Positive'])
print(report)

with open(os.path.join(outdir, 'classification_report_scoary.txt'), 'w') as f:
    f.write(report)

agg['labels'] = labels
agg.to_csv(outfile, index=False)
