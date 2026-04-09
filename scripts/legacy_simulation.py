"""
Legacy simulation methods extracted from Legacy-Simulation/SimulationMethods.py.
Used by runParamTraversal.py through legacy_tree_simulator.py.

Aliases at bottom match the names used in runParamTraversal.py:
    simulate        -> simulate_glrates_bit
    simulate_nodist -> simulate_glrates_nodist
    simulate_ctmp   -> simulate_glrates_ctmp
    simulate_norm   -> simulate_glrates_bit_norm
"""

from typing import List, Tuple, Set, Dict
import numpy as np
import pandas as pd
from ete3 import Tree
from joblib import Parallel, delayed, parallel_backend, Memory
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns


### Simulation Methods

def simulate_events(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml.
    Sums gains and losses to a total number of events then allocates each event to a
    branch on the tree with probability proportional to the length of the branch.
    For each trait only simulates on branches beyond a certain distance from the root.
    """
    # Preprocess and setup
    all_nodes = list(self.tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    total_events = self.gains
    losses = self.losses

    # Distance calculations
    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse():  # type: ignore
        if node in node_dists:
            continue
        node_dists[node] = node_dists[node.up] + node.dist
    node_df = pd.DataFrame.from_dict(node_dists, orient='index', columns=['total_dist'])
    node_df['dist'] = [node.dist for node in node_df.index]

    def get_nodes(dist):
        nodes = node_df[node_df['total_dist'] >= dist]
        nodes = nodes.assign(used_dist=np.minimum(nodes['total_dist'] - dist, nodes['dist']))
        bl = nodes['used_dist'].sum()
        p = nodes['used_dist'] / bl
        if any(p.isna()):
            return (None, None)
        node_index = [node_map[node] for node in nodes.index]
        return node_index, p

    def sim_events(trait, total_events, ap):
        a, p = ap
        if not a:
            return (None, trait)
        event_locs = np.apply_along_axis(
            lambda x: np.random.choice(a, size=x.shape[0], replace=False, p=p),
            arr=np.zeros((int(np.ceil(total_events[trait])), self.NUM_TRIALS)),
            axis=0
        )
        return (event_locs, trait)

    if self.parallel:
        branch_probabilities = [get_nodes(self.dists[trait]) for trait, events in enumerate(total_events)]
        all_event_locs = Parallel(n_jobs=-1, batch_size=100)(
            delayed(sim_events)(trait, total_events, branch_probabilities[trait])
            for trait in range(num_traits)
        )  # type: ignore
        for event_locs, trait in all_event_locs:  # type: ignore
            if event_locs is not None:
                sim[:, trait, :][
                    event_locs.flatten("F"),
                    np.repeat(np.arange(self.NUM_TRIALS), int(np.ceil(total_events[trait])))
                ] = True
    else:
        for trait, events in enumerate(total_events):
            a, p = get_nodes(self.dists[trait])
            if not a:
                continue
            event_locs = np.apply_along_axis(
                lambda x: np.random.choice(a, size=x.shape[0], replace=False, p=p),
                arr=np.zeros((int(np.ceil(events)), self.NUM_TRIALS)),
                axis=0
            )  # type: ignore
            sim[:, trait, :][
                event_locs.flatten("F"),
                np.repeat(np.arange(self.NUM_TRIALS), int(np.ceil(events)))
            ] = True

    # Lineage calculations
    for node in all_nodes:
        if node.up is None:
            root = self.root_states > 0
            sim[node_map[node], root, :] = True
            continue
        parent = sim[node_map[node.up], :, :]
        curr = sim[node_map[node], :, :]
        sim[node_map[node], :, :] = np.logical_xor(parent, curr)

    lineages = sim[[node_map[node] for node in self.tree], :, :]

    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [
            pd.DataFrame(sim[:, :, i],
                         index=[node.name for node in all_nodes],
                         columns=[self.mapping[str(j)] for j in range(num_traits)]
                         ).loc[self.tree.get_leaf_names()]
            for i in range(num)
        ]

    return res, trait_data, get_simulated_trees(5)


def simulate_glrates(self):
    """
    Simulates trees based off gains and losses inferred on observed data by pastml.
    Sums gains and losses to a total number of events then allocates each event to a
    branch on the tree with probability proportional to the length of the branch.
    """
    all_nodes = list(self.tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    gain_rates = self.gains / (self.gain_subsize * self.MULTIPLIER)
    loss_rates = self.losses / (self.loss_subsize * self.MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse():  # type: ignore
        if node in node_dists:
            continue
        node_dists[node] = node_dists[node.up] + node.dist

    for node in self.tree.traverse():  # type: ignore
        if node.up is None:
            root = self.root_states > 0
            sim[node_map[node], root, :] = True
            continue

        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]

        applicable_traits_gains = node_total_dist >= self.dists
        applicable_traits_losses = node_total_dist >= self.loss_dists
        gain_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)

        gain_events[applicable_traits_gains] = np.random.binomial(
            node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis],
            (applicable_traits_gains.sum(), self.NUM_TRIALS)
        ) > 0
        loss_events[applicable_traits_losses] = np.random.binomial(
            node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis],
            (applicable_traits_losses.sum(), self.NUM_TRIALS)
        ) > 0

        gain_mask = np.logical_not(np.logical_xor(parent, gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent, loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node], :, :] = parent.copy()
        sim[node_map[node], :, :][gain_events] = True
        sim[node_map[node], :, :][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]
    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [
            pd.DataFrame(sim[:, :, i],
                         index=[node.name for node in all_nodes],
                         columns=[self.mapping[str(j)] for j in range(num_traits)]
                         ).loc[self.tree.get_leaf_names()]
            for i in range(num)
        ]

    return res, trait_data, get_simulated_trees(5)


def simulate_glrates_ctmp(self):
    """
    CTMP simulation of trait evolution with gain/loss rates on a tree.
    Gains and losses occur as a Poisson process. Multiple events per branch possible.
    """
    all_nodes = list(self.tree.traverse())
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: idx for idx, node in enumerate(all_nodes)}

    gain_rates = self.gains / self.gain_subsize
    loss_rates = self.losses / self.loss_subsize
    gain_rates = np.nan_to_num(gain_rates)
    loss_rates = np.nan_to_num(loss_rates)

    sim[node_map[self.tree], :, :] = self.root_states[:, np.newaxis] > 0

    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse():  # type: ignore
        if node in node_dists:
            continue
        node_dists[node] = node_dists[node.up] + node.dist

    for node in self.tree.traverse():  # type: ignore
        if node.is_root():
            continue

        parent_idx = node_map[node.up]  # type: ignore
        curr_idx = node_map[node]
        dist = node.dist

        sim[curr_idx] = sim[parent_idx]

        for trait in range(num_traits):
            g = gain_rates[trait] if node_dists[node] > self.dists[trait] else 0
            l = loss_rates[trait] if node_dists[node] > self.loss_dists[trait] else 0
            for trial in range(self.NUM_TRIALS):
                state = sim[parent_idx, trait, trial]
                t = 0.0
                while t < dist:
                    rate = g if not state else l
                    if rate == 0:
                        break
                    wait_time = np.random.exponential(1 / rate)
                    if t + wait_time > dist:
                        break
                    t += wait_time
                    state = not state
                sim[curr_idx, trait, trial] = state

    lineages = sim[[node_map[node] for node in self.tree], :, :]
    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [
            pd.DataFrame(sim[:, :, i],
                         index=[n.name for n in all_nodes],
                         columns=[self.mapping[str(j)] for j in range(num_traits)]
                         ).loc[self.tree.get_leaf_names()]
            for i in range(num)
        ]

    return res, trait_data, get_simulated_trees(5)


def simulate_glrates_nodist(self):
    """
    Simulates trees based off gains and losses. Only simulates traits in self.pairs.
    Non-simulated traits remain constant (zeros except for root states).
    """
    all_nodes = list(self.tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    gain_rates = self.gains / (self.gain_subsize * self.MULTIPLIER)
    loss_rates = self.losses / (self.loss_subsize * self.MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    if hasattr(self, "pairs") and len(self.pairs) > 0:
        pairs_arr = np.array(self.pairs, dtype=int)
        traits_to_simulate = np.unique(pairs_arr.flatten())
        simulate_mask = np.zeros(num_traits, dtype=bool)
        simulate_mask[traits_to_simulate] = True
    else:
        simulate_mask = np.zeros(num_traits, dtype=bool)

    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse():  # type: ignore
        if node in node_dists:
            continue
        node_dists[node] = node_dists[node.up] + node.dist

    for node in self.tree.traverse():  # type: ignore
        if node.up is None:
            root = self.root_states > 0
            sim[node_map[node], root, :] = True
            continue

        parent = sim[node_map[node.up], :, :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]

        applicable_traits = simulate_mask
        gain_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)
        loss_events = np.zeros((num_traits, self.NUM_TRIALS), dtype=bool)

        if applicable_traits.any():
            idxs = np.nonzero(applicable_traits)[0]
            draws = np.random.binomial(
                node_dist_multiplier, gain_rates[idxs, np.newaxis],
                (len(idxs), self.NUM_TRIALS)
            ) > 0
            gain_events[idxs] = draws
            draws = np.random.binomial(
                node_dist_multiplier, loss_rates[idxs, np.newaxis],
                (len(idxs), self.NUM_TRIALS)
            ) > 0
            loss_events[idxs] = draws

        gain_mask = np.logical_not(np.logical_xor(parent, gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent, loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node], :, :] = parent.copy()
        sim[node_map[node], :, :][gain_events] = True
        sim[node_map[node], :, :][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]
    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [
            pd.DataFrame(sim[:, :, i],
                         index=[node.name for node in all_nodes],
                         columns=[self.mapping[str(j)] for j in range(num_traits)]
                         ).loc[self.tree.get_leaf_names()]
            for i in range(num)
        ]

    return res, trait_data, get_simulated_trees(5)


def simulate_distnorm(self):
    """
    Simulates trees based off gains and losses normalized by number of nodes.
    """
    all_nodes = list(self.tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    sim = np.zeros((num_nodes, num_traits, self.NUM_TRIALS), dtype=bool)
    node_map = {node: ind for ind, node in enumerate(all_nodes)}
    bl = 2 * len(self.tree) - 1

    gain_rates = self.gains / bl
    loss_rates = self.losses / bl
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    for node in self.tree.traverse():  # type: ignore
        if node.up is None:
            prev = self.obsdf_modified.mean()
            high_prev = list(prev[prev >= .5].index.astype(int))
            sim[node_map[node], high_prev, :] = True
            continue

        parent = sim[node_map[node.up], :, :]
        gain_events = np.random.binomial(1, gain_rates[:, np.newaxis], (len(gain_rates), self.NUM_TRIALS)) > 0
        loss_events = np.random.binomial(1, loss_rates[:, np.newaxis], (len(loss_rates), self.NUM_TRIALS)) > 0

        gain_mask = np.logical_not(np.logical_xor(parent, gain_events))
        gain_events[gain_mask] = False
        loss_mask = np.logical_xor(parent, loss_events)
        loss_events[loss_mask] = False

        sim[node_map[node], :, :] = parent.copy()
        sim[node_map[node], :, :][gain_events] = True
        sim[node_map[node], :, :][loss_events] = False

    lineages = sim[[node_map[node] for node in self.tree], :, :]
    res = compile_results(self, lineages)
    trait_data = calculate_trait_data(self, lineages, num_traits)

    def get_simulated_trees(num: int) -> list:
        return [
            pd.DataFrame(sim[:, :, i],
                         index=[node.name for node in all_nodes],
                         columns=[self.mapping[str(j)] for j in range(num_traits)]
                         ).loc[self.tree.get_leaf_names()]
            for i in range(num)
        ]

    return res, trait_data, get_simulated_trees(5)


def simulate_glrates_bit(self):
    """
    Bit-packed simulation using uint64 (64 trials per trait).
    Uses KDE-based p-value estimation via compile_results_KDE_bit_async.
    """
    all_nodes = list(self.tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    bits = 64
    nptype = np.uint64
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    self.NUM_TRIALS = bits

    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    gain_rates = self.gains / (self.gain_subsize * self.MULTIPLIER)
    loss_rates = self.losses / (self.loss_subsize * self.MULTIPLIER)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse():  # type: ignore
        if node in node_dists:
            continue
        node_dists[node] = node_dists[node.up] + node.dist

    print("Simulating Trees...")
    for node in self.tree.traverse():  # type: ignore
        if node.up is None:
            root = self.root_states > 0
            sim[node_map[node], root] = (1 << self.NUM_TRIALS) - 1
            continue

        parent = sim[node_map[node.up], :]
        node_dist_multiplier = node.dist * self.MULTIPLIER
        node_total_dist = node_dists[node]

        applicable_traits_gains = node_total_dist >= self.dists
        applicable_traits_losses = node_total_dist >= self.loss_dists
        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)
        gain_events[applicable_traits_gains] = np.packbits(
            (np.random.binomial(node_dist_multiplier, gain_rates[applicable_traits_gains, np.newaxis],
                                (applicable_traits_gains.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),
            axis=-1, bitorder='little'
        ).view(nptype).flatten()
        loss_events[applicable_traits_losses] = np.packbits(
            (np.random.binomial(node_dist_multiplier, loss_rates[applicable_traits_losses, np.newaxis],
                                (applicable_traits_losses.sum(), self.NUM_TRIALS)) > 0).astype(np.uint8),
            axis=-1, bitorder='little'
        ).view(nptype).flatten()

        updated_state = np.bitwise_or(parent, gain_events)
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))
        sim[node_map[node], :] = updated_state

    print("Completed Tree Simulation Successfully")
    lineages = sim[[node_map[node] for node in self.tree], :]
    res = compile_results_KDE_bit_async(self, lineages, bits=bits, nptype=nptype)
    return res


def simulate_glrates_bit_norm(self):
    """
    Bit-packed simulation normalized by number of nodes.
    Only simulates traits appearing in self.pairs.
    """
    all_nodes = list(self.tree.traverse())  # type: ignore
    num_nodes, num_traits = len(all_nodes), len(self.gains)
    bits = 64
    nptype = np.uint64
    sim = np.zeros((num_nodes, num_traits), dtype=nptype)
    self.NUM_TRIALS = bits

    node_map = {node: ind for ind, node in enumerate(all_nodes)}

    gain_rates = self.gains / len(all_nodes)
    loss_rates = self.losses / len(all_nodes)
    np.nan_to_num(gain_rates, copy=False)
    np.nan_to_num(loss_rates, copy=False)

    if hasattr(self, "pairs") and len(self.pairs) > 0:
        pairs_arr = np.array(self.pairs, dtype=int)
        traits_to_simulate = np.unique(pairs_arr.flatten())
        simulate_mask = np.zeros(num_traits, dtype=bool)
        simulate_mask[traits_to_simulate] = True
    else:
        simulate_mask = np.zeros(num_traits, dtype=bool)

    node_dists = {}
    node_dists[self.tree] = self.tree.dist or 0
    for node in self.tree.traverse():  # type: ignore
        if node in node_dists:
            continue
        node_dists[node] = node_dists[node.up] + node.dist

    print("Simulating Trees...")
    for node in self.tree.traverse():  # type: ignore
        if node.up is None:
            root = self.root_states > 0
            root_mask = np.zeros(num_traits, dtype=bool)
            root_mask[root] = True
            full_mask_value = (1 << self.NUM_TRIALS) - 1
            sim[node_map[node], root_mask] = full_mask_value
            continue

        parent = sim[node_map[node.up], :]
        node_total_dist = node_dists[node]

        applicable_traits_gains = (node_total_dist >= self.dists) & simulate_mask
        applicable_traits_losses = (node_total_dist >= self.loss_dists) & simulate_mask

        gain_events = np.zeros((num_traits), dtype=nptype)
        loss_events = np.zeros((num_traits), dtype=nptype)

        if applicable_traits_gains.any():
            idxs = np.nonzero(applicable_traits_gains)[0]
            draws = (np.random.binomial(1, gain_rates[idxs, np.newaxis],
                                        (len(idxs), self.NUM_TRIALS)) > 0).astype(np.uint8)
            packed = np.packbits(draws, axis=-1, bitorder='little')
            gain_events[idxs] = packed.view(nptype).flatten()

        if applicable_traits_losses.any():
            idxs = np.nonzero(applicable_traits_losses)[0]
            draws = (np.random.binomial(1, loss_rates[idxs, np.newaxis],
                                        (len(idxs), self.NUM_TRIALS)) > 0).astype(np.uint8)
            packed = np.packbits(draws, axis=-1, bitorder='little')
            loss_events[idxs] = packed.view(nptype).flatten()

        updated_state = np.bitwise_or(parent, gain_events)
        updated_state = np.bitwise_and(updated_state, np.bitwise_not(loss_events))
        sim[node_map[node], :] = updated_state

    print("Completed Tree Simulation Successfully")
    lineages = sim[[node_map[node] for node in self.tree], :]
    res = compile_results_KDE_bit_async(self, lineages, bits=bits, nptype=nptype)
    return res


### Result compilation functions

def compile_results(self, sim, obspairs=[]):
    if self.kde:
        if self.parallel:
            return compile_results_KDE_async(self, sim, obspairs)
        return compile_results_KDE(self, sim, obspairs)
    elif self.parallel:
        return compile_results_async(self, sim, obspairs)
    else:
        return compile_results_sync(self, sim, obspairs)


def compile_results_sync(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction",
                               "p-value_ant", "p-value_syn", "p-value", "significant",
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)
    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    syn = np.zeros(len(pairs))
    ant = np.zeros(len(pairs))
    means = np.zeros(len(pairs))
    medians = np.zeros(len(pairs))
    iqrs = np.zeros(len(pairs))

    for roll in range(tq.shape[-1]):
        rolled_tq = np.roll(tq, roll, axis=-1)
        rolled_cooc = self.pair_statistic(tp, rolled_tq)
        syn += np.sum(rolled_cooc >= obspairs[:, np.newaxis], axis=-1)
        ant += np.sum(rolled_cooc <= obspairs[:, np.newaxis], axis=-1)
        means += np.mean(rolled_cooc, axis=-1)
        medians += np.median(rolled_cooc, axis=-1)
        q75, q25 = np.percentile(rolled_cooc, [75, 25], axis=-1)
        iqrs += (q75 - q25)

    sim_trials = tq.shape[-1] ** 2
    means /= tq.shape[-1]
    medians /= tq.shape[-1]
    iqrs /= tq.shape[-1]

    pvals_ant = ant / sim_trials
    pvals_syn = syn / sim_trials
    pvals = np.minimum(pvals_syn, pvals_ant)
    directions = np.where(pvals_ant < pvals_syn, -1, 1)
    significants = pvals < 0.05
    effects = (medians - obspairs) / np.maximum(iqrs, 1)

    res['pair'] = [(p, q) for p, q in pairs]
    res['first'] = pairs[:, 0].tolist()
    res['second'] = pairs[:, 1].tolist()
    res['num_pair_trials'] = [sim_trials] * len(pairs)
    res['o_occ'] = obspairs.tolist()
    res['e_occ'] = means.tolist()
    res['p-value_ant'] = pvals_ant.tolist()
    res['p-value_syn'] = pvals_syn.tolist()
    res['p-value'] = pvals.tolist()
    res['direction'] = directions.tolist()
    res['significant'] = significants.tolist()
    res['median'] = medians.tolist()
    res['iqr'] = iqrs.tolist()
    res['effect size'] = effects.tolist()

    return pd.DataFrame.from_dict(res)


def compile_results_async2(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction",
                               "p-value_ant", "p-value_syn", "p-value", "significant",
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)
    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    def process_roll(roll):
        rolled_tq = np.roll(tq, roll, axis=-1)
        rolled_cooc = self.pair_statistic(tp, rolled_tq)
        syn = np.sum(rolled_cooc >= obspairs[:, np.newaxis], axis=-1)
        ant = np.sum(rolled_cooc <= obspairs[:, np.newaxis], axis=-1)
        mean = np.mean(rolled_cooc, axis=-1)
        median = np.median(rolled_cooc, axis=-1)
        q75, q25 = np.percentile(rolled_cooc, [75, 25], axis=-1)
        iqr = (q75 - q25)
        return syn, ant, mean, median, iqr

    with parallel_backend('loky', n_jobs=-1):
        results = Parallel(batch_size=10, return_as='generator')(
            delayed(process_roll)(roll) for roll in range(tq.shape[-1])
        )

    syn = np.sum([r[0] for r in results], axis=0)  # type: ignore
    ant = np.sum([r[1] for r in results], axis=0)  # type: ignore
    means = np.mean([r[2] for r in results], axis=0)  # type: ignore
    medians = np.mean([r[3] for r in results], axis=0)  # type: ignore
    iqrs = np.mean([r[4] for r in results], axis=0)  # type: ignore

    sim_trials = tq.shape[-1] ** 2
    pvals_ant = ant / sim_trials
    pvals_syn = syn / sim_trials
    pvals = np.minimum(pvals_syn, pvals_ant)
    directions = np.where(pvals_ant < pvals_syn, -1, 1)
    significants = pvals < 0.05
    effects = (medians - obspairs) / np.maximum(iqrs, 1)

    res['pair'] = [(p, q) for p, q in pairs]
    res['first'] = pairs[:, 0].tolist()
    res['second'] = pairs[:, 1].tolist()
    res['num_pair_trials'] = [sim_trials] * len(pairs)
    res['o_occ'] = obspairs.tolist()
    res['e_occ'] = means.tolist()
    res['p-value_ant'] = pvals_ant.tolist()
    res['p-value_syn'] = pvals_syn.tolist()
    res['p-value'] = pvals.tolist()
    res['direction'] = directions.tolist()
    res['significant'] = significants.tolist()
    res['median'] = medians.tolist()
    res['iqr'] = iqrs.tolist()
    res['effect size'] = effects.tolist()

    return pd.DataFrame.from_dict(res)


def compile_results_async(self, sim, obspairs=[]):

    def process_pair(ind, p, q, obs):
        tp, tq = sim[:, p, :], sim[:, q, :]
        ant, syn = 0, 0
        means = []
        medians = []
        iqrs = []
        for roll in range(tq.shape[1]):
            cooc = self.pair_statistic(tp, np.roll(tq, roll, axis=1))
            syn += np.sum(cooc >= obs)
            ant += np.sum(cooc <= obs)
            means.append(np.mean(cooc))
            medians.append(np.median(cooc))
            q75, q25 = np.percentile(cooc, [75, 25])
            iqr = q75 - q25
            iqrs.append(iqr)
        sim_trials = tq.shape[1] ** 2
        return (p, q, sim_trials, obs, syn, ant, means, medians, iqrs)

    obspairs = obspairs or self.obspairs
    pair_stats = Parallel(n_jobs=-1, batch_size=10, return_as='generator', verbose=10)(
        delayed(process_pair)(ind, p, q, obs)
        for ind, ((p, q), obs) in enumerate(zip(self.pairs, obspairs))
    )  # type: ignore

    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction",
                               "p-value_ant", "p-value_syn", "p-value", "significant",
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    for pair in pair_stats:
        p, q, sim_trials, obs, syn, ant, means, medians, iqrs = pair  # type: ignore
        update_result_dict2(res, p, q, sim_trials, obs, syn, ant, means, medians, iqrs)

    return pd.DataFrame.from_dict(res)


def calculate_trait_data(self, sim, num_traits):
    sums = (sim > 0).sum(axis=0)
    median_vals = np.median(sums, axis=1)
    q75, q25 = np.percentile(sums, [75, 25], axis=1)
    iqr_vals = q75 - q25

    return pd.DataFrame({
        "trait": np.arange(num_traits),
        "mean": sums.mean(axis=1),
        "std": sums.std(axis=1),
        "iqr": iqr_vals,
        "median": median_vals
    })


def update_result_dict(res, p, q, sim_trials, obs, cooc, cooccur_bool):
    res['pair'].append((p, q))
    res['first'].append(p)
    res['second'].append(q)
    res['num_pair_trials'].append(sim_trials)
    res['o_occ'].append(obs)
    res['e_occ'].append(np.mean(cooc[~np.isnan(cooc)]))
    pval_ant = np.sum(cooc <= obs) / sim_trials
    pval_syn = np.sum(cooc >= obs) / sim_trials
    res['p-value_ant'].append(pval_ant)
    res['p-value_syn'].append(pval_syn)
    res['p-value'].append(min(pval_syn, pval_ant))
    res['direction'].append(-1 if pval_ant < pval_syn else 1)
    res['significant'].append(res['p-value'][-1] < .05)


def update_result_dict2(res, p, q, sim_trials, obs, syn, ant, means, medians, iqrs):
    res['pair'].append((p, q))
    res['first'].append(p)
    res['second'].append(q)
    res['num_pair_trials'].append(sim_trials)
    res['o_occ'].append(obs)
    res['e_occ'].append(np.mean(means))
    pval_ant = ant / sim_trials
    pval_syn = syn / sim_trials
    res['p-value_ant'].append(pval_ant)
    res['p-value_syn'].append(pval_syn)
    res['p-value'].append(min(pval_syn, pval_ant))
    res['direction'].append(-1 if pval_ant < pval_syn else 1)
    res['significant'].append(res['p-value'][-1] < .05)
    median = np.mean(medians)
    iqr = np.mean(iqrs)
    res['median'].append(median)
    res['iqr'].append(iqr)
    res['effect size'].append((median - obs) / max(iqr, 1))


def compile_results_KDE(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction",
                               "p-value_ant", "p-value_syn", "p-value", "significant",
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    medians = np.zeros(len(pairs))
    iqrs = np.zeros(len(pairs))
    kde_pvals_ant = np.zeros(len(pairs))
    kde_pvals_syn = np.zeros(len(pairs))

    all_cooc = []
    for roll in range(tq.shape[-1]):
        rolled_tq = np.roll(tq, roll, axis=-1)
        rolled_cooc = self.pair_statistic(tp, rolled_tq)
        all_cooc.append(rolled_cooc)
    all_cooc = np.concatenate(all_cooc, axis=-1)

    for i in range(len(pairs)):
        noised = all_cooc[i] + np.random.normal(0, 1e-9, size=len(all_cooc[i]))
        kde = gaussian_kde(noised, bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1 * noised, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf, -x)
        kde_pvals_ant[i] = cdf_func_ant(obspairs[i])
        kde_pvals_syn[i] = cdf_func_syn(obspairs[i])
        medians[i] = np.median(all_cooc[i])
        q75, q25 = np.percentile(all_cooc[i], [75, 25])
        iqrs[i] = q75 - q25

    pvals = np.minimum(kde_pvals_syn, kde_pvals_ant)
    directions = np.where(kde_pvals_ant < kde_pvals_syn, -1, 1)
    significants = pvals < 0.05
    effects = (medians - obspairs) / np.maximum(iqrs, 1)

    res['pair'] = [(p, q) for p, q in pairs]
    res['first'] = pairs[:, 0].tolist()
    res['second'] = pairs[:, 1].tolist()
    res['num_pair_trials'] = [tq.shape[-1] ** 2] * len(pairs)
    res['o_occ'] = obspairs.tolist()
    res['e_occ'] = medians.tolist()
    res['p-value_ant'] = kde_pvals_ant.tolist()
    res['p-value_syn'] = kde_pvals_syn.tolist()
    res['p-value'] = pvals.tolist()
    res['direction'] = directions.tolist()
    res['significant'] = significants.tolist()
    res['median'] = medians.tolist()
    res['iqr'] = iqrs.tolist()
    res['effect size'] = effects.tolist()

    return pd.DataFrame.from_dict(res)


def compile_results_KDE_async(self, sim, obspairs=[]):
    res = {key: [] for key in ["pair", "first", "second", "num_pair_trials", "direction",
                               "p-value_ant", "p-value_syn", "p-value", "significant",
                               "e_occ", "o_occ", "median", "iqr", "effect size"]}

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    tp = sim[:, pairs[:, 0], :]
    tq = sim[:, pairs[:, 1], :]

    def compute_rolled_cooc(roll):
        rolled_tq = np.roll(tq, roll, axis=-1)
        return self.pair_statistic(tp, rolled_tq)

    all_cooc = Parallel(n_jobs=-1, verbose=10, batch_size=10)(
        delayed(compute_rolled_cooc)(roll) for roll in range(tq.shape[-1])
    )
    all_cooc = np.vstack(all_cooc)

    def compute_pair_stats(i):
        noised = all_cooc[i] + np.random.normal(0, 1e-9, size=len(all_cooc[i]))
        kde = gaussian_kde(noised, bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1 * noised, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf, -x)
        kde_pval_ant = cdf_func_ant(obspairs[i])
        kde_pval_syn = cdf_func_syn(obspairs[i])
        med = np.median(all_cooc[i])
        q75, q25 = np.percentile(all_cooc[i], [75, 25])
        iqr = q75 - q25
        return (i, kde_pval_ant, kde_pval_syn, med, iqr)

    results = Parallel(n_jobs=-1, batch_size=25, return_as='generator', verbose=10)(
        delayed(compute_pair_stats)(i) for i in range(len(pairs))
    )

    for i, kde_pval_ant, kde_pval_syn, med, iqr in results:
        pval = min(kde_pval_syn, kde_pval_ant)
        direction = -1 if kde_pval_ant < kde_pval_syn else 1
        significant = pval < 0.05
        effect_size = (med - obspairs[i]) / max(iqr, 1)

        res["pair"].append(tuple(pairs[i]))
        res["first"].append(pairs[i, 0])
        res["second"].append(pairs[i, 1])
        res["num_pair_trials"].append(tq.shape[-1] ** 2)
        res["o_occ"].append(obspairs[i])
        res["e_occ"].append(med)
        res["p-value_ant"].append(kde_pval_ant)
        res["p-value_syn"].append(kde_pval_syn)
        res["p-value"].append(pval)
        res["direction"].append(direction)
        res["significant"].append(significant)
        res["median"].append(med)
        res["iqr"].append(iqr)
        res["effect size"].append(effect_size)

    return pd.DataFrame.from_dict(res)


def compile_results_KDE_bit_async(
    self,
    sim: np.ndarray,
    obspairs: List[float] = [],
    batch_size: int = 1000,
    bits=64,
    nptype=np.uint64
) -> pd.DataFrame:
    """
    Compile KDE results asynchronously using parallel batch processing with bit-packed sim.
    """
    memory = Memory(location=None, verbose=0)
    res: Dict[str, List] = {
        "pair": [], "first": [], "second": [], "num_pair_trials": [],
        "direction": [], "p-value_ant": [], "p-value_syn": [], "p-value": [],
        "significant": [], "e_occ": [], "o_occ": [], "median": [], "iqr": [],
        "effect size": []
    }

    pairs = np.array(self.pairs)
    if not obspairs:
        obspairs = self.obspairs
    obspairs = np.array(obspairs)

    sim = np.asarray(sim, order="C")
    sim.setflags(write=False)

    def circular_bitshift_right(arr: np.ndarray, k: int) -> np.ndarray:
        k %= bits
        return np.bitwise_or(np.right_shift(arr, k), np.left_shift(arr, bits - k))

    def sum_all_bits(arr: np.ndarray) -> np.ndarray:
        bit_sums = np.zeros((bits, arr.shape[-1]), dtype=np.float64)
        for i in range(bits):
            bit_sums[i] = np.sum((arr >> i) & 1, axis=0, dtype=nptype)
        return bit_sums

    def compute_bitwise_cooc(tp: np.ndarray, tq: np.ndarray) -> np.ndarray:
        cooc_batch = []
        for k in range(bits):
            shifted = circular_bitshift_right(tq, k)
            a = sum_all_bits(tp & shifted) + 1e-2
            b = sum_all_bits(tp & ~shifted) + 1e-2
            c = sum_all_bits(~tp & shifted) + 1e-2
            d = sum_all_bits(~tp & ~shifted) + 1e-2
            cooc_batch.append(np.log((a * d) / (b * c)))
        return np.vstack(cooc_batch).T  # (batch_size, bits)

    def compute_kde_stats(observed_value: float, simulated_values: np.ndarray) -> Tuple[float, float, float, float]:
        kde = gaussian_kde(simulated_values, bw_method='silverman')
        cdf_func_ant = lambda x: kde.integrate_box_1d(-np.inf, x)
        kde_syn = gaussian_kde(-1 * simulated_values, bw_method='silverman')
        cdf_func_syn = lambda x: kde_syn.integrate_box_1d(-np.inf, -x)
        kde_pval_ant = cdf_func_ant(observed_value)
        kde_pval_syn = cdf_func_syn(observed_value)
        med = np.median(simulated_values)
        q75, q25 = np.percentile(simulated_values, [75, 25])
        iqr = q75 - q25
        return kde_pval_ant, kde_pval_syn, med, iqr

    def process_batch(index: int, sim_readonly: np.ndarray) -> Dict[str, List]:
        pair_batch = pairs[index: index + batch_size]
        tp = sim_readonly[:, pair_batch[:, 0]]
        tq = sim_readonly[:, pair_batch[:, 1]]
        batch_cooc = compute_bitwise_cooc(tp, tq)
        noised_batch_cooc = batch_cooc + np.random.normal(0, 1e-12, size=batch_cooc.shape)

        results = Parallel(n_jobs=-1, verbose=0, batch_size=25)(
            delayed(compute_kde_stats)(obspairs[index + i], noised_batch_cooc[i])
            for i in range(len(pair_batch))
        )

        kde_pvals_ant, kde_pvals_syn, medians, iqrs = map(np.array, zip(*results))

        batch_res = {
            "pair": [tuple(p) for p in pair_batch],
            "first": pair_batch[:, 0].tolist(),
            "second": pair_batch[:, 1].tolist(),
            "num_pair_trials": [sim_readonly.shape[1] ** 2] * len(pair_batch),
            "o_occ": obspairs[index: index + len(pair_batch)].tolist(),
            "e_occ": medians.tolist(),
            "median": medians.tolist(),
            "iqr": iqrs.tolist(),
            "p-value_ant": kde_pvals_ant.tolist(),
            "p-value_syn": kde_pvals_syn.tolist(),
            "p-value": np.minimum(kde_pvals_syn, kde_pvals_ant).tolist(),
            "direction": np.where(kde_pvals_ant < kde_pvals_syn, -1, 1).tolist(),
            "significant": (np.minimum(kde_pvals_syn, kde_pvals_ant) < 0.05).tolist(),
            "effect size": ((obspairs[index: index + len(pair_batch)] - medians) / np.maximum(iqrs * 1.349, 1)).tolist(),
        }
        return batch_res

    num_pairs = len(pairs)
    batch_indices = range(0, num_pairs, batch_size)

    print(f"Processing Batches, Total: {num_pairs // batch_size + 1}")
    batch_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_batch)(index, sim) for index in batch_indices
    )

    print("Aggregating Results...")
    for batch_res in batch_results:
        for key in res.keys():
            res[key].extend(batch_res[key])

    return pd.DataFrame.from_dict(res)


### Aliases used by runParamTraversal.py
simulate = simulate_glrates_bit
simulate_nodist = simulate_glrates_nodist
simulate_ctmp = simulate_glrates_ctmp
simulate_norm = simulate_glrates_bit_norm
