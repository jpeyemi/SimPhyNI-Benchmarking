##################################################
# TEST SNAKEFILE — SimPhyNI only                 #
# 10 pos pairs, 10 neg pairs, ~100 null pairs    #
# 2 fixed trees + 1 msprime tree, 1 effect size  #
##################################################

import sys
import os
import re

SCRIPTS_DIRECTORY = "./scripts"
sys.path.insert(0, SCRIPTS_DIRECTORY)

import pandas as pd
from ete3 import Tree
from generateTree import generate_msprime_tree

configfile: "config_test.yaml"
workdir: workflow.basedir

# ──────────────────────────────────────────────
# Wildcards
# ──────────────────────────────────────────────
FIXED_TREES  = config["fixed_trees"]
MSPRIME_MAIN = [f"msprime_{i}".replace(" ","") for i in range(config["num_msprime_main"])]
ALL_TREES    = list(FIXED_TREES) + MSPRIME_MAIN

EFFECT_SIZES = config["effect_sizes"]
ES_IDX       = list(range(len(EFFECT_SIZES)))

NUM_PAIRS    = config["num_pairs"]

SIMPHYNI_SCRIPTS     = os.path.join(os.path.dirname(workflow.snakefile), "SimPhyNI", "simphyni", "scripts")
BENCHMARK_SCRIPT     = os.path.join(os.path.dirname(workflow.snakefile), "SimPhyNI", "dev", "benchmark_reconstruction.py")
STABILITY_FAN_SCRIPT = os.path.join(os.path.dirname(workflow.snakefile), "SimPhyNI", "dev", "plot_stability_fan.py")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def reformat_string_for_filepath(s):
    replacements = {' ': '_', '\\': '', '/': '', ':': '', '*': '',
                    '?': '', '"': '', '<': '', '>': '', '|': '', '.': '_', '~': ''}
    for key, value in replacements.items():
        s = s.replace(key, value)
    s = re.sub(r'[^a-zA-Z0-9_.-]', '', s)
    return s

def process_tree(input_tree, output_tree):
    try:
        tree = Tree(input_tree)
    except Exception:
        tree = Tree(input_tree, 1)
    tree.name = 'root'
    for idx, node in enumerate(tree.iter_descendants("levelorder")):
        if not node.is_leaf() and node.name == "":
            node.name = f"internal_{idx}".replace(" ","")
    tree.write(format=1, outfile=output_tree)


# ──────────────────────────────────────────────
# rule all
# ──────────────────────────────────────────────
rule all:
    input:
        # "scripts/kde_model.pkl",
        # expand("test-results/{tree}/es{es}/simphyni_results.csv",              tree=ALL_TREES, es=ES_IDX),
        # expand("test-results/{tree}/es{es}/scoary_results.csv",                tree=ALL_TREES, es=ES_IDX),
        # expand("test-results/{tree}/es{es}/coinfinder_results.csv",            tree=ALL_TREES, es=ES_IDX),
        # expand("test-results/{tree}/es{es}/htreewas_terminal.csv",             tree=ALL_TREES, es=ES_IDX),
        # expand("test-results/{tree}/es{es}/fastlmm_results.csv",               tree=ALL_TREES, es=ES_IDX),
        expand("test-results/{tree}/es{es}/acr_benchmark/method_ranking.csv",  tree=ALL_TREES, es=ES_IDX),
        expand("test-results/{tree}/es{es}/acr_benchmark/stability_fans/.done",tree=ALL_TREES, es=ES_IDX),
        # expand("test-data/{tree}/es{es}/synth_stats.csv",                      tree=ALL_TREES, es=ES_IDX),


# ──────────────────────────────────────────────
# KDE build
# ──────────────────────────────────────────────
rule build_kde:
    input:
        acr  = "pastmlout_marginal.csv",
        tree = "inputs/species_trees/Escherichia_coli.nwk"
    output:
        kde = "scripts/kde_model.pkl"
    conda:
        "simphyni_dev"
    shell:
        "python scripts/build_kde.py "
        "  --acr  {input.acr} "
        "  --tree {input.tree} "
        "  --out  {output.kde}"


# ──────────────────────────────────────────────
# Tree preparation
# ──────────────────────────────────────────────
rule prepare_tree:
    output:
        "test-data/{tree}/tree.nwk"
    run:
        import os
        tree_name = wildcards.tree
        out_path  = output[0]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if tree_name in FIXED_TREES:
            import shutil
            shutil.copy(FIXED_TREES[tree_name], out_path)
        else:
            t = generate_msprime_tree(target_leaves=config["msprime_target_leaves"])
            t.write(outfile=out_path)


# ──────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────
rule generateData:
    input:
        treefile = "test-data/{tree}/tree.nwk",
        kde      = "scripts/kde_model.pkl"
    output:
        datafile    = "test-data/{tree}/es{es}/synthetic_data.csv",
        pair_labels = "test-data/{tree}/es{es}/pair_labels.csv"
    params:
        num_pairs   = NUM_PAIRS,
        effect_size = lambda wildcards: EFFECT_SIZES[int(wildcards.es)]
    run:
        import numpy as np
        from ete3 import Tree
        from makeSynthData import synth_mutual_4state_nosim, _load_kde, _compute_bl_stats
        from d_statistic import get_or_calibrate, get_null_distributions, compute_d_statistic
        import os

        t           = Tree(input.treefile, format=1)
        effect_size = params.effect_size

        kde, ecoli_mean_subsize = _load_kde(input.kde)
        bl_stats                = _compute_bl_stats(t)

        # D-statistic config
        d_cfg          = config.get("d_statistic", {})
        n_permutations = int(d_cfg.get("n_permutations", 999))
        d_low          = float(d_cfg.get("d_low_threshold", -0.05))
        d_high         = float(d_cfg.get("d_high_threshold",  0.05))

        # Precompute tree structure and BM vectors once per tree (cached by tree path)
        tree_structure, bm_leaf_vals = get_or_calibrate(
            input.treefile, t, n_permutations=n_permutations
        )

        os.makedirs(os.path.dirname(output.datafile), exist_ok=True)

        pos_pairs, neg_pairs = [], []
        for direction, storage in [(1, pos_pairs), (-1, neg_pairs)]:
            while len(storage) < params.num_pairs:
                result    = synth_mutual_4state_nosim(
                    direction, t, effect_size,
                    kde=kde, bl_stats=bl_stats,
                    ecoli_mean_subsize=ecoli_mean_subsize,
                )
                lineages, prev = result[0], result[1]
                if np.any(prev <= 0.05) or np.any(prev >= 0.95):
                    continue
                storage.append(lineages)

        leaf_names = [node.name for node in t.get_leaves()]
        n_pos      = params.num_pairs * 2
        n_neg      = params.num_pairs * 2
        n_traits   = n_pos + n_neg

        trait_arrays, pair_records = [], []
        for pair_idx, lineages in enumerate(pos_pairs):
            col_a, col_b = pair_idx * 2, pair_idx * 2 + 1
            trait_arrays += [lineages[:, 0], lineages[:, 1]]
            pair_records.append((col_a, col_b, 1))
        for pair_idx, lineages in enumerate(neg_pairs):
            col_a, col_b = n_pos + pair_idx * 2, n_pos + pair_idx * 2 + 1
            trait_arrays += [lineages[:, 0], lineages[:, 1]]
            pair_records.append((col_a, col_b, -1))

        rng             = np.random.default_rng()
        pos_col_indices = list(range(n_pos))
        neg_col_indices = list(range(n_pos, n_pos + n_neg))

        null_pool_A, null_pool_B, null_pool_C = [], [], []
        for pi in range(params.num_pairs):
            for pj in range(pi + 1, params.num_pairs):
                for ta in [pi*2, pi*2+1]:
                    for tb in [pj*2, pj*2+1]:
                        null_pool_A.append((ta, tb))
        for pi in range(params.num_pairs):
            for pj in range(pi + 1, params.num_pairs):
                for ta in [n_pos + pi*2, n_pos + pi*2+1]:
                    for tb in [n_pos + pj*2, n_pos + pj*2+1]:
                        null_pool_B.append((ta, tb))
        for pc in pos_col_indices:
            for nc in neg_col_indices:
                null_pool_C.append((pc, nc))

        n_null_target = params.num_pairs * 10
        n_per_pool    = n_null_target // 3
        remainder     = n_null_target - n_per_pool * 3

        def sample_pool(pool, n):
            pool = np.array(pool)
            idx  = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
            return pool[idx].tolist()

        null_sampled = (
            sample_pool(null_pool_A, n_per_pool) +
            sample_pool(null_pool_B, n_per_pool) +
            sample_pool(null_pool_C, n_per_pool + remainder)
        )
        for col_a, col_b in null_sampled:
            pair_records.append((col_a, col_b, 0))

        #perm               = rng.permutation(n_traits)
        col_names          = [f"synth_trait_{i}".replace(" ","") for i in range(n_traits)]
        #col_names_shuffled = [col_names[perm[i]] for i in range(n_traits)]
        data_matrix        = np.column_stack(trait_arrays)#[:, perm]

        leaf_order    = {name: i for i, name in enumerate(leaf_names)}
        ts_leaf_names = tree_structure["leaf_names"]

        def _compute_pair_d(col_a_idx, col_b_idx):
            t1   = trait_arrays[col_a_idx].astype(float)
            t2   = trait_arrays[col_b_idx].astype(float)
            ord1 = np.array([t1[leaf_order[n]] for n in ts_leaf_names], dtype=float)
            ord2 = np.array([t2[leaf_order[n]] for n in ts_leaf_names], dtype=float)
            rm1, bm1 = get_null_distributions(input.treefile, tree_structure, bm_leaf_vals, ord1.mean(), n_permutations)
            rm2, bm2 = get_null_distributions(input.treefile, tree_structure, bm_leaf_vals, ord2.mean(), n_permutations)
            d1   = compute_d_statistic(tree_structure, ord1, rm1, bm1)
            d2   = compute_d_statistic(tree_structure, ord2, rm2, bm2)
            finite = [v for v in (d1, d2) if not np.isnan(v)]
            return d1, d2, (float(max(finite)) if finite else float("nan"))

        def _assign_stratum(d_pair):
            if np.isnan(d_pair):    return "degenerate"
            elif d_pair <= d_low:   return "low_independence"
            elif d_pair > d_high:   return "high_independence"
            else:                   return "mid_independence"

        pair_label_rows = []
        for col_a_idx, col_b_idx, direction in pair_records:
            name_a = col_names[col_a_idx]
            name_b = col_names[col_b_idx]
            d1, d2, d_pair = _compute_pair_d(col_a_idx, col_b_idx)
            pair_label_rows.append({
                "trait1":      name_a,
                "trait2":      name_b,
                "direction":   direction,
                "d1":          d1,
                "d2":          d2,
                "d_statistic": d_pair,
                "d_stratum":   _assign_stratum(d_pair),
            })

        pd.DataFrame(data_matrix, index=leaf_names, columns=col_names).to_csv(output.datafile)
        pd.DataFrame(pair_label_rows).to_csv(output.pair_labels, index=False)


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────
rule verify_synthetic_data:
    input:
        datafile    = "test-data/{tree}/es{es}/synthetic_data.csv",
        pair_labels = "test-data/{tree}/es{es}/pair_labels.csv"
    output:
        stats    = "test-data/{tree}/es{es}/synth_stats.csv",
        prev_png = "test-data/{tree}/es{es}/synth_stats_prev.png",
        lor_png  = "test-data/{tree}/es{es}/synth_stats_lor.png"
    run:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = pd.read_csv(input.datafile, index_col=0)
        pl   = pd.read_csv(input.pair_labels)

        rows = []
        for _, row in pl.iterrows():
            t1, t2, direction = row["trait1"], row["trait2"], row["direction"]
            if t1 not in data.columns or t2 not in data.columns:
                continue
            x    = data[t1].values.astype(float)
            y    = data[t2].values.astype(float)
            n11  = np.sum((x == 1) & (y == 1))
            n10  = np.sum((x == 1) & (y == 0))
            n01  = np.sum((x == 0) & (y == 1))
            n00  = np.sum((x == 0) & (y == 0))
            lor  = float(np.log((n11 * n00) / (n10 * n01))) if n10 > 0 and n01 > 0 and n11 > 0 and n00 > 0 else float("nan")
            rows.append({"trait1": t1, "trait2": t2, "direction": direction,
                         "prev_trait1": x.mean(), "prev_trait2": y.mean(), "log_odds_ratio": lor})

        stats = pd.DataFrame(rows)
        stats.to_csv(output.stats, index=False)

        colors = {-1: "#e05c5c", 0: "#aaaaaa", 1: "#5c9ee0"}

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for direction in [-1, 0, 1]:
            sub = stats[stats["direction"] == direction]
            for ax, col in zip(axes, ["prev_trait1", "prev_trait2"]):
                vals = sub[col].dropna().values
                if len(vals) >= 2:
                    ax.violinplot([vals], positions=[direction], showmedians=True)
                elif len(vals) == 1:
                    ax.scatter([direction], vals, zorder=3)
        for ax, title in zip(axes, ["Trait 1 prevalence", "Trait 2 prevalence"]):
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["negative", "null", "positive"])
            ax.set_ylabel("Prevalence")
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output.prev_png, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for direction in [-1, 0, 1]:
            vals = stats[stats["direction"] == direction]["log_odds_ratio"].dropna().values
            if len(vals) >= 2:
                ax.violinplot([vals], positions=[direction], showmedians=True)
            elif len(vals) == 1:
                ax.scatter([direction], vals, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["negative", "null", "positive"])
        ax.set_ylabel("Log-odds ratio")
        ax.set_title("Association signal by direction")
        fig.tight_layout()
        fig.savefig(output.lor_png, dpi=150)
        plt.close(fig)


# ──────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────
rule reformat_csv:
    input:
        inp         = "test-data/{tree}/es{es}/synthetic_data.csv",
        pair_labels = "test-data/{tree}/es{es}/pair_labels.csv"
    output:
        out         = "test-formatting/{tree}/es{es}/reformated_systems.csv",
        pair_labels = "test-formatting/{tree}/es{es}/pair_labels.csv"
    run:
        import pandas as pd
        data     = pd.read_csv(input.inp, index_col=0)
        old_cols = list(data.columns)
        new_cols = [reformat_string_for_filepath(c) for c in old_cols]
        col_map  = dict(zip(old_cols, new_cols))
        data.columns = new_cols
        data[data > 0] = 1
        data.fillna(0, inplace=True)
        data.astype(int)
        data.to_csv(output.out)
        pl = pd.read_csv(input.pair_labels)
        pl["trait1"] = pl["trait1"].map(col_map)
        pl["trait2"] = pl["trait2"].map(col_map)
        pl.to_csv(output.pair_labels, index=False)


rule reformat_tree:
    input:
        inp = "test-data/{tree}/tree.nwk"
    output:
        out = "test-formatting/{tree}/reformated_tree.nwk"
    run:
        process_tree(input.inp, output.out)


# ──────────────────────────────────────────────
# Ancestral reconstruction
# ──────────────────────────────────────────────
rule ancestral_reconstruction:
    input:
        inputsFile = "test-formatting/{tree}/es{es}/reformated_systems.csv",
        tree       = "test-formatting/{tree}/reformated_tree.nwk"
    output:
        outfile = "test-pastml/{tree}/es{es}/pastmlout.csv"
    params:
        acr_script = SIMPHYNI_SCRIPTS + "/run_ancestral_reconstruction.py"
    threads: 16
    conda:
        "simphyni_dev"
    shell:
        "python {params.acr_script} "
        "  --inputs_file {input.inputsFile} "
        "  --tree_file {input.tree} "
        "  --output_csv {output.outfile} "
        "  --max_workers {threads} "
        "  --reconstruction all "


# ──────────────────────────────────────────────
# SimPhyNI
# ──────────────────────────────────────────────
rule simphyni:
    input:
        pastml      = "test-pastml/{tree}/es{es}/pastmlout.csv",
        systems     = "test-formatting/{tree}/es{es}/reformated_systems.csv",
        tree        = "test-formatting/{tree}/reformated_tree.nwk",
        pair_labels = "test-formatting/{tree}/es{es}/pair_labels.csv"
    output:
        outfile = "test-results/{tree}/es{es}/simphyni_results.csv"
    threads: 16
    conda:
        "simphyni_dev"
    shell:
        "python scripts/runSimPhyNI.py "
        "  --pastml {input.pastml} "
        "  --systems {input.systems} "
        "  --tree {input.tree} "
        "  --pair_labels {input.pair_labels} "
        "  --outfile {output.outfile} "


# ──────────────────────────────────────────────
# Scoary
# ──────────────────────────────────────────────
rule runScoary:
    input:
        systems     = "test-formatting/{tree}/es{es}/reformated_systems.csv",
        tree        = "test-formatting/{tree}/reformated_tree.nwk",
        pair_labels = "test-formatting/{tree}/es{es}/pair_labels.csv"
    output:
        outfile = "test-results/{tree}/es{es}/scoary_results.csv"
    conda:
        "scoary-2"
    shell:
        "python scripts/runScoary.py "
        "  --systems {input.systems} "
        "  --tree {input.tree} "
        "  --pair_labels {input.pair_labels} "
        "  --outfile {output.outfile}"


# ──────────────────────────────────────────────
# Coinfinder
# ──────────────────────────────────────────────
rule runCoinfinder:
    input:
        systems     = "test-formatting/{tree}/es{es}/reformated_systems.csv",
        tree        = "test-formatting/{tree}/reformated_tree.nwk",
        pair_labels = "test-formatting/{tree}/es{es}/pair_labels.csv"
    output:
        outfile = "test-results/{tree}/es{es}/coinfinder_results.csv"
    threads: 16
    conda:
        "coinfinder"
    shell:
        "python scripts/runCoinfinder.py "
        "  --systems {input.systems} "
        "  --tree {input.tree} "
        "  --pair_labels {input.pair_labels} "
        "  --outfile {output.outfile} "
        "  --threads {threads}"


# ──────────────────────────────────────────────
# TreeWAS
# ──────────────────────────────────────────────
rule run_htreewas_analysis:
    input:
        traits      = "test-formatting/{tree}/es{es}/reformated_systems.csv",
        tree        = "test-formatting/{tree}/reformated_tree.nwk",
        pair_labels = "test-formatting/{tree}/es{es}/pair_labels.csv"
    output:
        terminal     = "test-results/{tree}/es{es}/htreewas_terminal.csv",
        simultaneous = "test-results/{tree}/es{es}/htreewas_simultaneous.csv",
        subsequent   = "test-results/{tree}/es{es}/htreewas_subsequent.csv"
    conda:
        "r-env"
    shell:
        "Rscript scripts/run_treewas_homo.R "
        "  --tree {input.tree} "
        "  --traits {input.traits} "
        "  --pair_labels {input.pair_labels} "
        "  --terminal {output.terminal} "
        "  --simultaneous {output.simultaneous} "
        "  --subsequent {output.subsequent}"


# ──────────────────────────────────────────────
# FastLMM
# ──────────────────────────────────────────────
rule run_fastlmm_phy:
    input:
        tree = "test-formatting/{tree}/reformated_tree.nwk"
    output:
        outfile = "test-results/{tree}/es{es}/fastlmm/phylogeny_similarity.tsv"
    params:
        working_dir = "test-results/{tree}/es{es}/fastlmm"
    conda:
        "pyseer"
    shell:
        "mkdir -p {params.working_dir} && "
        "python scripts/phylogeny_distance.py --lmm {input.tree} > {output.outfile}"


rule run_fastlmm:
    input:
        traits      = "test-formatting/{tree}/es{es}/reformated_systems.csv",
        tree        = "test-formatting/{tree}/reformated_tree.nwk",
        phylo       = "test-results/{tree}/es{es}/fastlmm/phylogeny_similarity.tsv",
        pair_labels = "test-formatting/{tree}/es{es}/pair_labels.csv"
    output:
        outfile = "test-results/{tree}/es{es}/fastlmm_results.csv"
    params:
        working_dir = "test-results/{tree}/es{es}/fastlmm"
    conda:
        "pyseer"
    shell:
        "python scripts/runFastLMM.py "
        "  --trait_file {input.traits} "
        "  --working_dir {params.working_dir} "
        "  --kinship {input.phylo} "
        "  --pair_labels {input.pair_labels} "
        "  --outfile {output.outfile}"


# ──────────────────────────────────────────────
# ACR Benchmark
# ──────────────────────────────────────────────
rule acr_benchmark:
    input:
        data        = "test-data/{tree}/es{es}/synthetic_data.csv",
        tree        = "test-formatting/{tree}/reformated_tree.nwk",
        pair_labels = "test-formatting/{tree}/es{es}/pair_labels.csv"
    output:
        ranking    = "test-results/{tree}/es{es}/acr_benchmark/method_ranking.csv",
        trajectory = "test-results/{tree}/es{es}/acr_benchmark/stability_trajectory.csv"
    params:
        outdir            = "test-results/{tree}/es{es}/acr_benchmark",
        benchmark_script  = BENCHMARK_SCRIPT,
        n_stability       = config["benchmark_n_stability"],
        n_stability_iters = config["benchmark_n_stability_iters"]
    threads: 16
    conda:
        "simphyni_dev"
    shell:
        "python {params.benchmark_script} "
        "  --tree {input.tree} "
        "  --annotations {input.data} "
        "  --output {params.outdir} "
        "  --max_workers {threads} "
        "  --eval_sim_accuracy "
        "  --sim_accuracy_n 50 "
        "  --n_stability {params.n_stability} "
        "  --n_stability_iters {params.n_stability_iters} "
        "  --eval_pr "
        "  --known_pairs {input.pair_labels} "
        "  --no_legacy "


# ──────────────────────────────────────────────
# Stability Fan
# ──────────────────────────────────────────────
rule plot_stability_fan:
    input:
        trajectory = "test-results/{tree}/es{es}/acr_benchmark/stability_trajectory.csv"
    output:
        done = "test-results/{tree}/es{es}/acr_benchmark/stability_fans/.done"
    params:
        outdir     = "test-results/{tree}/es{es}/acr_benchmark/stability_fans",
        fan_script = STABILITY_FAN_SCRIPT
    conda:
        "simphyni_dev"
    shell:
        "python {params.fan_script} "
        "  --trajectory {input.trajectory} "
        "  --output {params.outdir} "
        "&& touch {output.done}"
