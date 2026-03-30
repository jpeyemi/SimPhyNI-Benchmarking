import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_recall_curve, auc, roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from legacy_tree_simulator import TreeSimulator
from legacy_simulation import simulate, simulate_nodist, simulate_ctmp, simulate_norm
from simphyni.Simulation.pair_statistics import pair_statistics as PairStatistics
from scipy import stats
import statsmodels.stats.multitest as sm

# Initialize paths
tree_path = sys.argv[-2]
pastml_path = sys.argv[-4]
obs_data_path = sys.argv[-3]
outfile = sys.argv[-1]
os.makedirs(os.path.dirname(outfile), exist_ok=True)


simulation_methods = [simulate,simulate_nodist,simulate_ctmp,simulate_norm]
pair_statistics = [PairStatistics._log_odds_ratio_statistic, PairStatistics._log_add_ratio_statistic, PairStatistics._treewas_statistic, PairStatistics._jaccard_index_statistic, PairStatistics._mutual_information_statistic, PairStatistics.z_statistic, PairStatistics.count_statistic]
kde_indexes = [0,1]
thresholds = [0.05, 0.01, 0.001]
correction_conditions = [('fdr_bh', 0.01), ('fdr_by', 0.01), ('bonferroni', 0.01)]
significance_conditions = thresholds + correction_conditions

# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=['Method', 'Statistic', 'Threshold', 'Bonferroni', 'Precision_Negative', 
                                   'Recall_Negative', 'F1_Negative', 'Precision_Positive', 
                                   'Recall_Positive', 'F1_Positive', 'Accuracy', 'AUC_ROC_Negative', 'PR_AUC_Negative', 'AUC_ROC_Positive', 'PR_AUC_Positive', 'FDR_Negative', 'FPR_Negative', 'FDR_Positive', 'FPR_Positive'])

def evaluate(name, Sim):
    neg_auc, neg_pr = None, None
    pos_auc, pos_pr = None, None
    for condition in significance_conditions:
        if type(condition) == tuple:
            significance_threshold = condition[1]
            correction = condition[0]
        else:
            significance_threshold = condition
            correction = False
        # Get top results with current settings
        res = Sim.get_top_results(correction, top=int(len(Sim.obsdf.columns)), alpha = significance_threshold).sort_index()
        predicted_directions = res['direction']
        pvalues = res['p-value']
        labels = [0]*3000 + [-1]*300 + [1]*300 
        res['labels'] = labels
        if not neg_auc:
            neg_auc, neg_pr = AUCs(res['labels'],pvalues,predicted_directions,-1,str(name) + '_neg')
            pos_auc, pos_pr = AUCs(res['labels'],pvalues,predicted_directions,1,str(name) + '_pos')

        # Filter out non-significant predictions
        filtered_predictions = [pred if pval < significance_threshold else 0 for pred, pval in zip(predicted_directions, pvalues)]

        filtered_predictions = np.array(filtered_predictions)
        labels = np.array(labels)

        # Generate the classification report
        precision, recall, f1, _ = precision_recall_fscore_support(labels, filtered_predictions, average=None, labels=[-1, 1])
        accuracy = np.mean(filtered_predictions == labels)

        fdrp, fdrn = calcFDR(labels,pvalues,predicted_directions,significance_threshold)
        fprp, fprn = calcFPR(labels,pvalues,predicted_directions,significance_threshold)


        print('Params ', name, 'Threshold:', significance_threshold, 'Correction', correction)
        print(classification_report(labels, filtered_predictions, target_names=['Negative', 'None', 'Positive']))
        
        # Save results into the DataFrame
        results_df.loc[len(results_df)] = [name[0], name[1], significance_threshold, correction,
                                           precision[0], recall[0], f1[0],
                                           precision[1], recall[1], f1[1],
                                           accuracy, neg_auc, neg_pr, pos_auc, pos_pr, fdrn, fprn, fdrp, fprp]

def calcFDR(labels, pvalues, directions, threshold=0.05):
    """
    Calculate the False Discovery Rate (FDR) separately for positive and negative directions.
    
    Parameters:
    - labels: array-like of true labels
    - pvalues: array-like of p-values
    - directions: array-like of predicted directions (-1, 0, or 1)
    - threshold: float, p-value threshold to consider a positive discovery
    
    Returns:
    - fdr_positive: float, False Discovery Rate for positive direction (1) at the specified threshold
    - fdr_negative: float, False Discovery Rate for negative direction (-1) at the specified threshold
    """
    # Convert to numpy arrays if not already
    labels = np.array(labels)
    pvalues = np.array(pvalues)
    directions = np.array(directions)

    # Filter predictions below threshold
    significant = pvalues <= threshold
    
    # Separate FDR calculations for positive (1) and negative (-1) directions
    # Positive direction (1)
    predicted_positives = (directions == 1) & significant
    true_positives = np.sum((labels == 1) & predicted_positives)
    false_positives = np.sum((labels != 1) & predicted_positives)
    fdr_positive = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    # Negative direction (-1)
    predicted_negatives = (directions == -1) & significant
    true_negatives = np.sum((labels == -1) & predicted_negatives)
    false_negatives = np.sum((labels != -1) & predicted_negatives)
    fdr_negative = false_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0.0

    return fdr_positive, fdr_negative

def calcFPR(labels, pvalues, directions, threshold=0.05):
    """
    Calculate the False Positive Rate (FPR) separately for positive and negative directions.
    
    Parameters:
    - labels: array-like of true labels
    - pvalues: array-like of p-values
    - directions: array-like of predicted directions (-1, 0, or 1)
    - threshold: float, p-value threshold to consider a positive discovery
    
    Returns:
    - fpr_positive: float, False Positive Rate for positive direction (1) at the specified threshold
    - fpr_negative: float, False Positive Rate for negative direction (-1) at the specified threshold
    """
    # Convert to numpy arrays if not already
    labels = np.array(labels)
    pvalues = np.array(pvalues)
    directions = np.array(directions)

    # Filter predictions below threshold
    significant = pvalues <= threshold

    # Positive direction (1)
    predicted_positives = (directions == 1) & significant
    false_positives_pos = np.sum((labels != 1) & predicted_positives)
    total_negatives_pos = np.sum(labels != 1)
    fpr_positive = false_positives_pos / total_negatives_pos if total_negatives_pos > 0 else 0.0

    # Negative direction (-1)
    predicted_negatives = (directions == -1) & significant
    false_positives_neg = np.sum((labels != -1) & predicted_negatives)
    total_negatives_neg = np.sum(labels != -1)
    fpr_negative = false_positives_neg / total_negatives_neg if total_negatives_neg > 0 else 0.0

    return fpr_positive, fpr_negative

def AUCs(labels, pvalues, predicted_directions, dir, name):
    binary_labels = (labels == dir).astype(int)  # Only consider positive class for AUC
    pv = pvalues.copy()
    pv[predicted_directions == (dir * -1)] = 1 - pv[predicted_directions == (dir * -1)]
    min_pvalue = 1e-300
    capped_pvalues = np.maximum(pv, min_pvalue)
    # Generate ROC curve values
    fpr, tpr, thresholds = roc_curve(binary_labels, -np.log10(capped_pvalues))  # Use -log10(pvalues) for better AUC calculation
    auc_score = roc_auc_score(binary_labels, -np.log10(capped_pvalues))

    # os.makedirs(os.path.join(roc_path, name), exist_ok = True)
    # # Plot ROC curve
    # plt.figure(figsize=(5, 5))
    # plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='#1f77b4')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # roc_file_path = os.path.join(roc_path, name, 'ROC.png')
    # plt.savefig(roc_file_path)
    # plt.close() 

    precision, recall, thresholds = precision_recall_curve(binary_labels, -np.log10(capped_pvalues))

    # Calculate AUC for the precision-recall curve
    pr_auc = auc(recall, precision)

    # # Plot precision-recall curve
    # plt.figure(figsize=(5, 5))
    # plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}', color='red')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc='lower left')
    # plt.grid(True)
    # pr_file_path = os.path.join(roc_path, name, 'PR_Curve.png')
    # plt.savefig(pr_file_path)
    # plt.close()
    return auc_score, pr_auc



# Initialize simulations
Sims = {(method.__name__, stat.__name__): (method,stat) 
            for stat in pair_statistics 
                for method in simulation_methods}

for sim in Sims:
    if type(Sims[sim]) == tuple:
        Sim : TreeSimulator = TreeSimulator(tree = tree_path, 
                    pastmlfile=pastml_path, 
                    obsdatafile=obs_data_path)

        Sim.initialize_simulation_parameters(pair_statistic= Sims[sim][1],collapse_theshold=0.00, single_trait=True,prevalence_threshold=0.00, kde = Sims[sim][1] in [pair_statistics[j] for j in kde_indexes])
        Sim.set_trials(64)

        refs = list(range(0,int(len(Sim.obsdf.columns)),2))
        pairs = []
        for i in refs:
            pairs.extend([(i,i+j) for j in range(1,2)])
        Sim.pairs, Sim.obspairs = Sim._get_pair_data2(Sim.obsdf_modified,pairs)

        if Sims[sim][0] == simulation_methods[0] and Sims[sim][1] == pair_statistics[0]:
            bit = True
            sim_func = None
        else: 
            bit = False
            sim_func = Sims[sim][0]

        Sim.run_simulation(parallel=True, simulation_function= sim_func, bit = bit, norm = False)
        Sims[sim] = Sim # type: ignore

# Evaluate simulations
for i in Sims:
    evaluate(i, Sims[i])


# Save the results to a CSV file
results_df.to_csv(outfile, index=False)
