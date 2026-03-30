##########################
# SNAKEFILE FOR SimPhyNI #
# Benchmarking Pipeline  #
##########################

""" PRE-SNAKEMAKE """

import sys
import os
import importlib.resources

SCRIPTS_DIRECTORY = "./scripts"
sys.path.insert(0, SCRIPTS_DIRECTORY)

import pandas as pd
from ete3 import Tree
from generateTree import generate_msprime_tree

configfile: "config.yaml"

# ----------------------
# Tree wildcards
# ----------------------
FIXED_TREES = config["fixed_trees"]            # {name: path}
MSPRIME_MAIN = [f"msprime_{i}" for i in range(config["num_msprime_main"])]
ALL_TREES = list(FIXED_TREES) + MSPRIME_MAIN   # 12 fixed + 3 msprime = 15 total

EFFECT_SIZES = config["effect_sizes"]           # [3, 2, 1, 0.75, 0.5, 0.25]
ES_IDX = list(range(len(EFFECT_SIZES)))         # [0, 1, 2, 3, 4, 5]

BENCH_TREES = [f"bench_{i}" for i in range(config["num_msprime_benchmark"])]

NUM_PAIRS = config["num_pairs"]

# Path to SimPhyNI scripts
# SIMPHYNI_SCRIPTS = os.path.join(os.path.dirname(__file__), "SimPhyNI", "simphyni", "scripts")
# BENCHMARK_SCRIPT = os.path.join(os.path.dirname(__file__), "SimPhyNI", "dev", "benchmark_reconstruction.py")

SIMPHYNI_SCRIPTS     = os.path.join(os.path.dirname(workflow.snakefile), "SimPhyNI", "simphyni", "scripts")
BENCHMARK_SCRIPT     = os.path.join(os.path.dirname(workflow.snakefile), "SimPhyNI", "dev", "benchmark_reconstruction.py")
STABILITY_FAN_SCRIPT = os.path.join(os.path.dirname(workflow.snakefile), "SimPhyNI", "dev", "plot_stability_fan.py")

# ----------------------
# Helper functions
# ----------------------
def reformat_string_for_filepath(s):
    replacements = {' ': '_', '\\': '', '/': '', ':': '', '*': '',
                    '?': '', '"': '', '<': '', '>': '', '|': '', '.': '_', '~': ''}
    for key, value in replacements.items():
        s = s.replace(key, value)
    import re
    s = re.sub(r'[^a-zA-Z0-9_.-]', '', s)
    return s

def reformat_columns(input_csv, output_csv):
    data = pd.read_csv(input_csv, index_col=0)
    data.columns = [reformat_string_for_filepath(col) for col in data.columns]
    data[data > 0] = 1
    data.fillna(0, inplace=True)
    data.astype(int)
    data.to_csv(output_csv)

def process_tree(input_tree, output_tree):
    try:
        tree = Tree(input_tree)
    except Exception:
        tree = Tree(input_tree, 1)
    tree.name = 'root'
    for idx, node in enumerate(tree.iter_descendants("levelorder")):
        if not node.is_leaf() and node.name == "":
            node.name = f"internal_{idx}"
    tree.write(format=1, outfile=output_tree)


''' SNAKEMAKE '''

rule all:
    input:
        expand("2-Results/{tree}/es{es}/simphyni_results.csv",     tree=ALL_TREES, es=ES_IDX),
        expand("2-Results/{tree}/es{es}/htreewas_terminal.csv",    tree=ALL_TREES, es=ES_IDX),
        expand("2-Results/{tree}/es{es}/scoary_results.csv",       tree=ALL_TREES, es=ES_IDX),
        expand("2-Results/{tree}/es{es}/pagel_results.csv",        tree=ALL_TREES, es=ES_IDX),
        expand("2-Results/{tree}/es{es}/fastlmm_results.csv",      tree=ALL_TREES, es=ES_IDX),
        expand("2-Results/{tree}/es{es}/spydrpick_results.csv",    tree=ALL_TREES, es=ES_IDX),
        expand("2-Results/{tree}/es{es}/coinfinder_results.csv",   tree=ALL_TREES, es=ES_IDX),
        # expand("2-Results/{tree}/es{es}/goldfinder_terminal_results.csv",     tree=ALL_TREES, es=ES_IDX),
        # expand("2-Results/{tree}/es{es}/goldfinder_simultaneous_results.csv", tree=ALL_TREES, es=ES_IDX),
        # expand("2-Results/{tree}/es{es}/goldfinder_subsequent_results.csv",   tree=ALL_TREES, es=ES_IDX),
        # expand("2-Results/{tree}/es{es}/goldfinder_coinfinder_results.csv",   tree=ALL_TREES, es=ES_IDX),
        expand("benchmark-acr/{bench}/acr_benchmark/method_ranking.csv",       bench=BENCH_TREES),
        expand("benchmark-acr/{bench}/acr_benchmark/stability_fans/.done",     bench=BENCH_TREES),


rule prepare_tree:
    output:
        "0-GenerateTrees/{tree}/tree.nwk"
    run:
        import os
        tree_name = wildcards.tree
        out_path = output[0]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if tree_name in FIXED_TREES:
            os.system(f"cp {FIXED_TREES[tree_name]} {out_path}")
        else:
            # msprime tree
            t = generate_msprime_tree(target_leaves=config["msprime_target_leaves"])
            t.write(outfile=out_path)


rule prepare_benchmark_tree:
    output:
        "benchmark-acr/{bench}/tree.nwk"
    run:
        import os
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        t = generate_msprime_tree(target_leaves=config["msprime_target_leaves"])
        t.write(outfile=output[0])


rule generateData:
    input:
        treefile = "0-GenerateTrees/{tree}/tree.nwk"
    output:
        datafile = "0-GenerateTrees/{tree}/es{es}/synthetic_data.csv"
    params:
        num_pairs = NUM_PAIRS,
        effect_size = lambda wildcards: EFFECT_SIZES[int(wildcards.es)]
    run:
        import numpy as np
        from ete3 import Tree
        from makeSynthData import synth_mutual_4state_nosim, _load_kde, _compute_bl_stats
        import os

        t = Tree(input.treefile, format=1)
        effect_size = params.effect_size

        # Pre-compute shared resources once per tree
        kde = _load_kde('scripts/kde_model.pkl')
        bl_stats = _compute_bl_stats(t)

        os.makedirs(os.path.dirname(output.datafile), exist_ok=True)

        synthetic_datas = []
        for dir in [0, -1, 1]:
            data = []
            target = params.num_pairs * (10 if dir == 0 else 1)
            while len(data) < target:
                pair = synth_mutual_4state_nosim(dir, t, effect_size, kde=kde, bl_stats=bl_stats)
                if np.any(pair[1] <= 0.05) or np.any(pair[1] >= 0.95):
                    continue
                data.append(pair)
            synthetic_datas.extend(data)

        synthetic_data = np.hstack([d[0][:, :] for d in synthetic_datas])
        df = pd.DataFrame(
            synthetic_data,
            index=[node.name for node in synthetic_datas[0][-1]],
            columns=[f"synth_trait_{i}" for i in range(synthetic_data.shape[1])]
        )
        df.to_csv(output.datafile)


rule generate_benchmark_data:
    input:
        treefile = "benchmark-acr/{bench}/reformated_tree.nwk"
    output:
        datafile = "benchmark-acr/{bench}/synthetic_data.csv"
    params:
        num_pairs = NUM_PAIRS
    run:
        import numpy as np
        from ete3 import Tree
        from makeSynthData import synth_mutual_4state_nosim, _load_kde, _compute_bl_stats
        import os

        t = Tree(input.treefile, format=1)
        kde = _load_kde('scripts/kde_model.pkl')
        bl_stats = _compute_bl_stats(t)

        os.makedirs(os.path.dirname(output.datafile), exist_ok=True)

        synthetic_datas = []
        for dir in [0, -1, 1]:
            data = []
            target = params.num_pairs * (10 if dir == 0 else 1)
            while len(data) < target:
                pair = synth_mutual_4state_nosim(dir, t, 1.0, kde=kde, bl_stats=bl_stats)
                if np.any(pair[1] <= 0.05) or np.any(pair[1] >= 0.95):
                    continue
                data.append(pair)
            synthetic_datas.extend(data)

        synthetic_data = np.hstack([d[0][:, :] for d in synthetic_datas])
        df = pd.DataFrame(
            synthetic_data,
            index=[node.name for node in synthetic_datas[0][-1]],
            columns=[f"synth_trait_{i}" for i in range(synthetic_data.shape[1])]
        )
        df.to_csv(output.datafile)


# generate_known_pairs: builds the fixed known_pairs.csv shared by all acr_benchmark runs.
# Trait layout (0-indexed, 2 traits per pair):
#   dir=0  (null):   num_pairs*10 pairs → traits 0 .. num_pairs*20-1
#   dir=-1 (antag):  num_pairs pairs    → traits num_pairs*20 .. num_pairs*22-1
#   dir=+1 (synerg): num_pairs pairs    → traits num_pairs*22 .. num_pairs*24-1
rule generate_known_pairs:
    output:
        pairs = "benchmark-acr/known_pairs.csv"
    params:
        num_pairs = NUM_PAIRS
    run:
        import pandas as pd, os
        os.makedirs("benchmark-acr", exist_ok=True)
        null_pairs = params.num_pairs * 10      # 3000
        neg_start  = null_pairs * 2             # trait index 6000
        pos_start  = neg_start + params.num_pairs * 2  # trait index 6600
        rows = []
        for k in range(params.num_pairs):       # 300 antagonistic pairs
            t1 = neg_start + 2 * k
            rows.append({"T1": f"synth_trait_{t1}", "T2": f"synth_trait_{t1 + 1}", "direction": -1})
        for k in range(params.num_pairs):       # 300 synergistic pairs
            t1 = pos_start + 2 * k
            rows.append({"T1": f"synth_trait_{t1}", "T2": f"synth_trait_{t1 + 1}", "direction": 1})
        pd.DataFrame(rows).to_csv(output.pairs, index=False)


rule reformat_csv:
    input:
        inp = "0-GenerateTrees/{tree}/es{es}/synthetic_data.csv"
    output:
        out = "0-formatting/{tree}/es{es}/reformated_systems.csv"
    run:
        reformat_columns(input.inp, output.out)


rule reformat_tree:
    input:
        inp = "0-GenerateTrees/{tree}/tree.nwk"
    output:
        out = "0-formatting/{tree}/reformated_tree.nwk"
    run:
        process_tree(input.inp, output.out)


rule reformat_benchmark_tree:
    input:
        inp = "benchmark-acr/{bench}/tree.nwk"
    output:
        out = "benchmark-acr/{bench}/reformated_tree.nwk"
    run:
        process_tree(input.inp, output.out)


rule ancestral_reconstruction:
    input:
        inputsFile = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree       = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        outfile = "1-PastML-api/{tree}/es{es}/pastmlout.csv"
    params:
        acr_script = SIMPHYNI_SCRIPTS + "/run_ancestral_reconstruction.py"
    threads: 64
    conda:
        "simphyni_dev"
    shell:
        "python {params.acr_script} "
        "  --inputs_file {input.inputsFile} "
        "  --tree_file {input.tree} "
        "  --output_csv {output.outfile} "
        "  --max_workers {threads} "
        "  --uncertainty both "


rule simphyni:
    input:
        pastml  = "1-PastML-api/{tree}/es{es}/pastmlout.csv",
        systems = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree    = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        outfile = "2-Results/{tree}/es{es}/simphyni_results.csv"
    threads: 64
    conda:
        "simphyni_dev"
    shell:
        "python scripts/runSimPhyNI.py "
        "  --pastml {input.pastml} "
        "  --systems {input.systems} "
        "  --tree {input.tree} "
        "  --outfile {output.outfile} "


rule run_htreewas_analysis:
    input:
        traits = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree   = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        terminal     = "2-Results/{tree}/es{es}/htreewas_terminal.csv",
        simultaneous = "2-Results/{tree}/es{es}/htreewas_simultaneous.csv",
        subsequent   = "2-Results/{tree}/es{es}/htreewas_subsequent.csv"
    conda:
        'envs/R.yaml'
    shell:
        "Rscript scripts/run_treewas_homo.R "
        "  --tree {input.tree} "
        "  --traits {input.traits} "
        "  --terminal {output.terminal} "
        "  --simultaneous {output.simultaneous} "
        "  --subsequent {output.subsequent}"


rule runScoary:
    input:
        systems = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree    = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        annotation = "2-Results/{tree}/es{es}/scoary_results.csv"
    conda:
        'envs/scoary2.yaml'
    shell:
        "python scripts/runScoary.py {input.systems} {input.tree} {output.annotation}"


rule runPagel:
    input:
        traits = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree   = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        outfile = "2-Results/{tree}/es{es}/pagel_results.csv"
    conda:
        'envs/R.yaml'
    shell:
        "Rscript scripts/run_pagel.R "
        "  --tree {input.tree} "
        "  --traits {input.traits} "
        "  --outfile {output.outfile}"


rule run_fastlmm_phy:
    input:
        tree = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        outfile = "2-Results/{tree}/es{es}/fastlmm/phylogeny_similarity.tsv"
    params:
        working_dir = "2-Results/{tree}/es{es}/fastlmm"
    conda:
        'envs/pyseer.yaml'
    shell:
        "mkdir -p {params.working_dir} && "
        "python scripts/phylogeny_distance.py --lmm {input.tree} > {output.outfile}"


rule run_fastlmm:
    input:
        traits = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree   = "0-formatting/{tree}/reformated_tree.nwk",
        phylo  = "2-Results/{tree}/es{es}/fastlmm/phylogeny_similarity.tsv"
    output:
        outfile = "2-Results/{tree}/es{es}/fastlmm_results.csv"
    params:
        working_dir = "2-Results/{tree}/es{es}/fastlmm"
    conda:
        'envs/pyseer.yaml'
    shell:
        "python scripts/runFastLMM.py "
        "  --trait_file {input.traits} "
        "  --working_dir {params.working_dir} "
        "  --kinship {input.phylo} "
        "  --outfile {output.outfile}"


rule acr_benchmark:
    input:
        data        = "benchmark-acr/{bench}/synthetic_data.csv",
        tree        = "benchmark-acr/{bench}/reformated_tree.nwk",
        known_pairs = "benchmark-acr/known_pairs.csv"
    output:
        ranking    = "benchmark-acr/{bench}/acr_benchmark/method_ranking.csv",
        trajectory = "benchmark-acr/{bench}/acr_benchmark/stability_trajectory.csv"
    params:
        outdir             = "benchmark-acr/{bench}/acr_benchmark",
        benchmark_script   = BENCHMARK_SCRIPT,
        n_stability        = config["benchmark_n_stability"],
        n_stability_iters  = config["benchmark_n_stability_iters"]
    threads: 64
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
        "  --known_pairs {input.known_pairs} "
        "  --no_legacy "


rule plot_stability_fan:
    input:
        trajectory = "benchmark-acr/{bench}/acr_benchmark/stability_trajectory.csv"
    output:
        done = "benchmark-acr/{bench}/acr_benchmark/stability_fans/.done"
    params:
        outdir          = "benchmark-acr/{bench}/acr_benchmark/stability_fans",
        fan_script      = STABILITY_FAN_SCRIPT
    conda:
        "simphyni_dev"
    shell:
        "python {params.fan_script} "
        "  --trajectory {input.trajectory} "
        "  --output {params.outdir} "
        "&& touch {output.done}"


rule runSpydrPick:
    input:
        systems = "0-formatting/{tree}/es{es}/reformated_systems.csv"
    output:
        outfile = "2-Results/{tree}/es{es}/spydrpick_results.csv"
    conda:
        'envs/spydrpick.yaml'
    shell:
        "python scripts/runSpydrPick.py "
        "  --systems {input.systems} "
        "  --outfile {output.outfile}"


rule runCoinfinder:
    input:
        systems = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree    = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        outfile = "2-Results/{tree}/es{es}/coinfinder_results.csv"
    threads: 16
    conda:
        'envs/coinfinder.yaml'
    shell:
        "python scripts/runCoinfinder.py "
        "  --systems {input.systems} "
        "  --tree {input.tree} "
        "  --outfile {output.outfile} "
        "  --threads {threads} "
        "  --batch-size 100"


rule runGOLDfinder:
    input:
        systems = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree    = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        terminal     = "2-Results/{tree}/es{es}/goldfinder_terminal_results.csv",
        simultaneous = "2-Results/{tree}/es{es}/goldfinder_simultaneous_results.csv",
        subsequent   = "2-Results/{tree}/es{es}/goldfinder_subsequent_results.csv",
        coinfinder   = "2-Results/{tree}/es{es}/goldfinder_coinfinder_results.csv",
    shell:
        "conda run -n goldfinder python scripts/runGOLDfinder.py "
        "  --systems {input.systems} "
        "  --tree {input.tree} "
        "  --outdir 2-Results/{wildcards.tree}/es{wildcards.es}"


# Optional: parameter traversal (not in rule all by default)
rule ParamTraversal:
    input:
        pastml  = "1-PastML-api/{tree}/es{es}/pastmlout.csv",
        systems = "0-formatting/{tree}/es{es}/reformated_systems.csv",
        tree    = "0-formatting/{tree}/reformated_tree.nwk"
    output:
        annotation = "2-Results/{tree}/es{es}/paramtraversal.csv"
    conda:
        "simphyni_dev"
    shell:
        "python scripts/runParamTraversal.py "
        "  {input.pastml} {input.systems} {input.tree} {output.annotation}"
