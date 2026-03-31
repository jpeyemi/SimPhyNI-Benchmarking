#%%
from ete3 import Tree
import numpy as np
import random
import math
import msprime

def generate_coalescence_tree(target_leaves=500, initial_branch_length=100, decay_rate=0.5):
    tree = Tree()
    root = tree.add_child(name="root", dist=initial_branch_length)
    leaves = [root]
    counter = 0
    
    while len(leaves) < target_leaves:
        leaf = random.choice(leaves)
        decay_len = leaf.dist * math.exp(-decay_rate)
        # decay_len = leaf.dist * math.exp(-decay_rate)
        branch_len1 = max(np.random.normal(decay_len,decay_len**.5), 0.001)
        branch_len2 = max(np.random.normal(decay_len,decay_len**.5), 0.001)
        child1 = leaf.add_child(name=f"n{counter}", dist=branch_len1)
        counter += 1
        child2 = leaf.add_child(name=f"n{counter}", dist=branch_len2)
        counter += 1
        leaves.remove(leaf)
        leaves.extend([child1, child2])
    for node in tree.traverse():
        node.dist = node.dist * tree.get_distance(node, topology_only=True)
    return tree

def delta_transform(node, delta):
    if not node.is_root():
        # Apply the delta to the depth
        node.dist = node.dist ** delta
    for child in node.children:
        delta_transform(child, delta)

def generate_msprime_tree(target_leaves=500, population_size = 10):
    # return Tree(generate_structured_tree())
    tree_sequence = msprime.sim_ancestry(
        samples=target_leaves,
        population_size=population_size,
        recombination_rate=0,
        sequence_length=1,
        ploidy = 1,
    )
    newick = tree_sequence.first().newick()
    tree = Tree(newick)
    # delta_transform(tree,0.5)
    return tree


def generate_structured_tree(target_leaves=500, n_lineages=5,
                              ancestral_size=5000, lineage_size=1000,
                              max_split_time=5000, growth_rate=0.001):
    """
    Pectinate multi-lineage bacterial tree.

    n_lineages populations split sequentially from an Ancestral population at
    equally-spaced times. Within-clade exponential growth (growth_rate > 0)
    creates shallow terminal branches that mimic bacterial clonal expansion,
    giving an imbalanced, pectinate topology closer to real bacterial phylogenies
    than the previous symmetric 2-population model.
    """
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size=ancestral_size)

    lineage_names = [f"Lin{i}" for i in range(n_lineages)]
    split_times = np.linspace(max_split_time / n_lineages, max_split_time, n_lineages)

    for name in lineage_names:
        demography.add_population(name=name, initial_size=lineage_size)
        demography.add_population_parameters_change(
            time=0, growth_rate=growth_rate, population=name
        )

    for name, t_split in zip(lineage_names, split_times):
        demography.add_population_split(
            time=t_split, derived=[name], ancestral="Ancestral"
        )

    samples_per_lineage = target_leaves // n_lineages
    samples = {name: samples_per_lineage for name in lineage_names}

    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        recombination_rate=0,
        sequence_length=1,
        ploidy=1,
    )
    return ts.first().newick()

# #%%
# coalescence_tree = generate_coalescence_tree(500,100,.5)
# print(coalescence_tree)

# # %%
# coalescence_tree.write(outfile='simtree.nwk', format=1)
# # %%
