#%%
import pandas as pd
import os
import subprocess
import numpy as np
import argparse
import glob
from sklearn.metrics import classification_report
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import tempfile

#%%
parser = argparse.ArgumentParser(description="Run PySEER analysis with FastLMM.")
parser.add_argument("--trait_file", required=True, help="Path to the trait annotation file.")
parser.add_argument("--working_dir", required=True, help="Directory to store results.")
parser.add_argument("--kinship_file", required=True, help="Path to the kinship similarity matrix.")
parser.add_argument("--outfile", required=True, help="Final Output filename")

args = parser.parse_args()

trait_file = args.trait_file
working_dir = args.working_dir
kinship_file = args.kinship_file
outfile = args.outfile

os.makedirs(working_dir, exist_ok=True)

#%%
df = pd.read_csv(trait_file, index_col=0)
columns = df.columns

#%%
from io import StringIO

def run_pyseer_pair(col1, col2):
    # Create temp files for inputs
    with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_pheno, \
         tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_pres:

        df[[col2]].to_csv(temp_pheno.name, sep='\t', header=False)
        df[[col1]].reset_index().rename(lambda x: "Gene" if x == 'index' else x).transpose().to_csv(temp_pres.name, sep='\t', header=False)

        pyseer_cmd = [
            'pyseer',
            '--phenotypes', temp_pheno.name,
            '--pres', temp_pres.name,
            '--similarity', kinship_file,
            '--lmm'
        ]

        # Run pyseer and capture STDOUT directly
        result = subprocess.run(pyseer_cmd, text=True, capture_output=True, check=True)

        # Read pyseer output directly into pandas
        pyseer_output = pd.read_csv(StringIO(result.stdout), sep='\t')
        pyseer_output['Trait'] = col1
        return pyseer_output

# def run_pyseer_pair(col1, col2):
#     temp_phenotype_file = os.path.join(working_dir, f'phenotypes_{col1}_{col2}.txt')
#     temp_file = os.path.join(working_dir, f'pres_{col1}_{col2}.txt')
#     output_file = os.path.join(working_dir, f'pyseer_{col1}_{col2}.txt')

#     df[[col1]].to_csv(temp_phenotype_file, sep='\t', header=False)
#     pd.DataFrame(df[[col2]]).reset_index().rename(lambda x: "Gene" if x == 'index' else x).transpose().to_csv(temp_file, sep='\t', header=False)

#     pyseer_cmd = [
#         'pyseer',
#         '--phenotypes', temp_phenotype_file,
#         '--pres', temp_file,
#         '--similarity', kinship_file,
#         '--lmm',
#     ]

#     try:
#         result = subprocess.run(pyseer_cmd, text=True, check=True, stdout=subprocess.PIPE)
#         with open(output_file, 'w') as f:
#             f.write(result.stdout)
#         os.remove(temp_file)
#         os.remove(temp_phenotype_file)
#         print(f"Completed: {col1}, {col2}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error for {col1}, {col2}: {e.stderr}",flush=True)

#%%
trait_pairs = [(columns[i], columns[i + 1]) for i in range(0, len(columns) - 1, 2)]

results = []

with ProcessPoolExecutor() as executor:
    futures = {executor.submit(run_pyseer_pair, c1, c2): (c1, c2) for c1, c2 in trait_pairs}
    for future in as_completed(futures):
        c1, c2 = futures[future]
        try:
            results.append(future.result())
            print(f"Completed: {c1}, {c2}")
        except Exception as e:
            print(f"Error in {c1}, {c2}: {e}")

agg = pd.concat(results, ignore_index=True)
expected_traits = [f'synth_trait_{i}' for i in range(0,7200,2)]
present_traits = set(agg['Trait'])
missing_traits = set(expected_traits) - present_traits
if missing_traits:
    missing_df = pd.DataFrame({
        'Trait': list(missing_traits),
        'beta': 0,
        'lrt-pvalue': 1,
    })
    agg = pd.concat([agg, missing_df], ignore_index=True)

agg['order'] = agg['Trait'].str.extract(r'(\d+)$').astype(int)
agg.sort_values(by='order', inplace=True)
agg.to_csv(args.outfile, index=False)

# Evaluate Performance
agg['pred'] = agg['beta'].apply(lambda x: 1 if x > 0 else -1)
labels = np.array([0]*3000 + [-1]*300 + [1]*300)
filtered_preds = np.where(agg['lrt-pvalue'] < 0.05, agg['pred'], 0)

from sklearn.metrics import classification_report
report = classification_report(labels, filtered_preds, target_names=['Negative', 'None', 'Positive'])
print(report)

with open(os.path.join(os.path.dirname(args.outfile), 'classification_report_fastlmm.txt'), 'w') as f:
    f.write(report)
    

# Run in parallel
# with ProcessPoolExecutor() as executor:
#     executor.map(lambda pair: run_pyseer_pair(*pair), trait_pairs)

# with ProcessPoolExecutor() as executor:
#     futures = {executor.submit(run_pyseer_pair, col1, col2): (col1, col2) for col1, col2 in trait_pairs}
#     for future in as_completed(futures):
#         col1, col2 = futures[future]
#         try:
#             future.result()  # Will raise if the process errored
#         except Exception as e:
#             print(f"Exception in ({col1}, {col2}): {e}", flush=True)

# print("All PySEER FastLMM processes completed.")

# #%%
# def aggregate(path_prefix, output_csv):
#     files = glob.glob(f"{path_prefix}*.txt")
#     dataframes = []
#     for file in files:
#         try:
#             df = pd.read_csv(file, sep='\t', encoding='utf-8')
#             df['source_file'] = os.path.basename(file)
#             dataframes.append(df)
#         except Exception as e:
#             print(f"Skipping {file}: {e}")

#     if dataframes:
#         expected_traits = [f'synth_trait_{i}' for i in range(0,7200,2)]
#         aggregated_df = pd.concat(dataframes, ignore_index=True)
#         aggregated_df['Trait'] = aggregated_df['variant'].astype(str).str.extract(r'(\d+)$').astype(int) - 1
#         aggregated_df['Trait'] = 'synth_trait_' + aggregated_df['Trait'].astype(str)
#         aggregated_df = aggregated_df[['Trait'] + [col for col in aggregated_df.columns if col != 'Trait']]
#         present_traits = set(aggregated_df['Trait'])
#         missing_traits = set(expected_traits) - present_traits
#         if missing_traits:
#             missing_df = pd.DataFrame({
#                 'Trait': list(missing_traits),
#                 'beta': 0,
#                 'lrt-pvalue': 1,
#             })
#             aggregated_df = pd.concat([aggregated_df, missing_df], ignore_index=True)
#         aggregated_df['order'] = aggregated_df['Trait'].astype(str).str.extract(r'(\d+)$').astype(int)
#         aggregated_df = aggregated_df.sort_values(by='order', ascending = True)
#         aggregated_df.to_csv(output_csv, index=False)
#         print(f"Aggregated {len(files)} files into {output_csv}")
#     else:
#         print("No valid files found.")

# aggregate(os.path.join(working_dir,"pyseer"), outfile)

# #%%
# res = pd.read_csv(outfile)
# res['pred'] = [1 if i > 0 else -1 for i in res['beta']]
# labels = [0]*3000 + [-1]*300 + [1]*300
# predictions = np.array(res['pred'])
# pvalues = res['lrt-pvalue'].fillna(1)
# labels = np.array(labels)
# significance_threshold = 0.05
# filtered_predictions = np.array([pred if pval < significance_threshold else 0 for pred, pval in zip(predictions, pvalues)])

# report = classification_report(labels, filtered_predictions, target_names=['Negative', 'None', 'Positive'])
# print('Params: ')
# print(report)

# with open(os.path.join(os.path.dirname(outfile),'classification_report_fastlmm.txt'), 'w') as file:
#     file.write("Classification Report:\n")
#     file.write(report)

# res['labels'] = labels
# res.to_csv(outfile)

# # Cleanup
# if os.path.exists(working_dir):
#     shutil.rmtree(working_dir)
#     print(f'Deleted the directory: {working_dir}')
# # %%
