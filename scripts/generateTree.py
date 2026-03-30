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


# #%%
# coalescence_tree = generate_coalescence_tree(500,100,.5)
# print(coalescence_tree)

# # %%
# coalescence_tree.write(outfile='simtree.nwk', format=1)
# # %%
