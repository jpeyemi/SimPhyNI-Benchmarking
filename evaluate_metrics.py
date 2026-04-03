import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
from statsmodels.stats.multitest import multipletests
import sys


def detect_file_type(df):
    if 'fq*ep' in df.columns and 'empirical_p' in df.columns:
        return 'scoary'
    if 'pval_by' in df.columns and 'p-value' in df.columns:
        return 'simphyni'
    if 'p_value' in df.columns and 'direction' in df.columns:
        return 'coinfinder'
    return None


def evaluate_simphyni(df, alpha):
    eps = np.finfo(float).eps
    df['neg_log_pval'] = -np.log10(df['p-value'].clip(lower=eps))

    results = {}
    for target_dir, label_name in [(1, "Direction 1 (Positive)"), (-1, "Direction -1 (Negative)")]:
        y_true = (df['label'] == target_dir).astype(int)

        is_significant = df['pval_by'] < alpha
        matches_predicted_dir = df['direction'] == target_dir
        y_pred = (is_significant & matches_predicted_dir).astype(int)

        y_score = df['neg_log_pval'] * matches_predicted_dir.astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec_curve, prec_curve)

        results[label_name] = {'Precision': precision, 'Recall': recall, 'PR_AUC': pr_auc}
    return results


def evaluate_coinfinder(df, alpha):
    eps = np.finfo(float).eps

    _, fdr_pvals, _, _ = multipletests(df['p_value'].fillna(1).values, method='fdr_by')
    df = df.copy()
    df['threshold_p']   = fdr_pvals
    df['neg_log_pval']  = -np.log10(df['p_value'].fillna(1.0).clip(lower=eps))

    results = {}
    for target_dir, label_name in [(1, "Direction 1 (Positive)"), (-1, "Direction -1 (Negative)")]:
        y_true = (df['label'] == target_dir).astype(int)

        is_significant        = df['threshold_p'] < alpha
        matches_predicted_dir = df['direction'] == target_dir
        y_pred  = (is_significant & matches_predicted_dir).astype(int)
        y_score = df['neg_log_pval'] * matches_predicted_dir.astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec_curve, prec_curve)

        results[label_name] = {'Precision': precision, 'Recall': recall, 'PR_AUC': pr_auc}

    return results


def evaluate_scoary(df, alpha, use_fisher):
    eps = np.finfo(float).eps

    # Infer predicted direction from odds_ratio
    df['direction'] = np.where(df['odds_ratio'] >= 1, 1, -1)

    if use_fisher:
        score_col     = 'fisher_p'
        # FDR-BY correction of fisher_p for the threshold
        _, fdr_pvals, _, _ = multipletests(df['fisher_p'].fillna(1).values, method='fdr_by')
        df['threshold_p'] = fdr_pvals
        threshold_label = 'fisher_p (FDR-BY corrected)'
    else:
        score_col     = 'fq*ep'
        _, fdr_pvals, _, _ = multipletests(df['empirical_p'].fillna(1).values, method='fdr_by')
        df['threshold_p'] = fdr_pvals
        threshold_label = 'empirical_p (FDR-BY corrected)'

    df['neg_log_score'] = -np.log10(df[score_col].fillna(1.0).clip(lower=eps))

    results = {}
    for target_dir, label_name in [(1, "Direction 1 (Positive)"), (-1, "Direction -1 (Negative)")]:
        y_true = (df['label'] == target_dir).astype(int)

        is_significant      = df['threshold_p'] < alpha
        matches_predicted_dir = df['direction'] == target_dir
        y_pred = (is_significant & matches_predicted_dir).astype(int)

        y_score = df['neg_log_score'] * matches_predicted_dir.astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec_curve, prec_curve)

        results[label_name] = {'Precision': precision, 'Recall': recall, 'PR_AUC': pr_auc}

    return results, score_col, threshold_label


def evaluate_predictions(file_path, alpha, use_fisher, pair_labels_path=None):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

    file_type = detect_file_type(df)
    if file_type is None:
        print("Error: Could not detect file type. Expected SimPhyNI or Scoary output.")
        sys.exit(1)

    print(f"\nDetected file type: {file_type.upper()}")

    # Merge d_stratum from pair_labels if provided
    if pair_labels_path is not None:
        try:
            pl = pd.read_csv(pair_labels_path)
            if "d_stratum" in pl.columns:
                left_on = ["T1", "T2"] if "T1" in df.columns else ["trait1", "trait2"]
                df = df.merge(
                    pl[["trait1", "trait2", "d_statistic", "d_stratum"]],
                    left_on=left_on,
                    right_on=["trait1", "trait2"],
                    how="left",
                )
        except Exception as e:
            print(f"Warning: could not load pair labels for stratification: {e}")

    if file_type == 'simphyni':
        if use_fisher:
            print("Warning: --fisher has no effect on SimPhyNI files.")
        results = evaluate_simphyni(df, alpha)
        score_desc     = "-log10(p-value)"
        threshold_desc = f"pval_by < {alpha}"
    elif file_type == 'coinfinder':
        if use_fisher:
            print("Warning: --fisher has no effect on Coinfinder files.")
        results = evaluate_coinfinder(df, alpha)
        score_desc     = "-log10(p_value)"
        threshold_desc = f"p_value (FDR-BY corrected) < {alpha}"
    else:
        results, score_col, threshold_label = evaluate_scoary(df, alpha, use_fisher)
        score_desc     = f"-log10({score_col})"
        threshold_desc = f"{threshold_label} < {alpha}"

    print(f"Score column   : {score_desc}")
    print(f"Threshold      : {threshold_desc}")
    print(f"\n--- Evaluation Results ---")
    for dir_name, metrics in results.items():
        print(f"\n** {dir_name} **")
        print(f"  Precision : {metrics['Precision']:.4f}")
        print(f"  Recall    : {metrics['Recall']:.4f}")
        print(f"  PR AUC    : {metrics['PR_AUC']:.4f}")

    # Corrected aggregate: low_independence TP pairs relabeled as non-TP
    if "d_stratum" in df.columns:
        df_corrected = df.copy()
        is_low_tp = (df_corrected["d_stratum"] == "low_independence") & (df_corrected["label"] != 0)
        df_corrected.loc[is_low_tp, "label"] = 0
        n_relabeled = int(is_low_tp.sum())
        print(f"\n--- Corrected Aggregate (low_independence TPs relabeled as non-TP, n={n_relabeled}) ---")
        if file_type == "simphyni":
            c_results = evaluate_simphyni(df_corrected, alpha)
        elif file_type == "coinfinder":
            c_results = evaluate_coinfinder(df_corrected, alpha)
        else:
            c_results, _, _ = evaluate_scoary(df_corrected, alpha, use_fisher)
        for dir_name, m in c_results.items():
            print(f"\n** {dir_name} **")
            print(f"  Precision : {m['Precision']:.4f}")
            print(f"  Recall    : {m['Recall']:.4f}")
            print(f"  PR AUC    : {m['PR_AUC']:.4f}")

    # Stratified output (only when d_stratum column is present)
    if "d_stratum" in df.columns:
        print("\n--- Stratified Results by D-Stratum ---")
        for stratum in ["low_independence", "mid_independence", "high_independence", "degenerate"]:
            sub = df[df["d_stratum"] == stratum].copy()
            if len(sub) == 0:
                continue
            print(f"\n[{stratum}]  (n={len(sub)} pairs)")
            if file_type == "simphyni":
                s_results = evaluate_simphyni(sub, alpha)
            elif file_type == "coinfinder":
                s_results = evaluate_coinfinder(sub, alpha)
            else:
                s_results, _, _ = evaluate_scoary(sub, alpha, use_fisher)
            for dir_name, m in s_results.items():
                print(f"  ** {dir_name} **")
                print(f"    Precision : {m['Precision']:.4f}")
                print(f"    Recall    : {m['Recall']:.4f}")
                print(f"    PR AUC    : {m['PR_AUC']:.4f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Precision, Recall, and PR AUC for SimPhyNI or Scoary output files."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to a SimPhyNI or Scoary results CSV."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance threshold (default: 0.01)."
    )
    parser.add_argument(
        "--fisher",
        action="store_true",
        help="(Scoary only) Use fisher_p for PR AUC and FDR-BY-corrected fisher_p for precision/recall."
    )
    parser.add_argument(
        "--pair-labels",
        type=str,
        default=None,
        dest="pair_labels",
        help="Path to pair_labels.csv (with d_stratum column) for stratified evaluation by phylogenetic independence."
    )

    args = parser.parse_args()
    evaluate_predictions(args.input_file, args.alpha, args.fisher, pair_labels_path=args.pair_labels)
