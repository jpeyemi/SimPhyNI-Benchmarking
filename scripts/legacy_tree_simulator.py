"""
Legacy TreeSimulator extracted from Legacy-Simulation/tree_simulator.py.
Used by runParamTraversal.py for parameter traversal benchmarking.

Imports updated to:
  - use legacy_simulation instead of SimulationMethods
  - use simphyni.Simulation.pair_statistics instead of PairStatistics
  - use simphyni.Simulation.Utils instead of Utils
  - guard plotly imports (optional visualization only)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from ete3 import Tree, TreeNode
from legacy_simulation import *
from itertools import cycle, combinations
from scipy.stats import fisher_exact
import random
import statsmodels.stats.multitest as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Union, Literal, List, Tuple, Set, Dict

from simphyni.Simulation.pair_statistics import pair_statistics as PairStatistics

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


class TreeSimulator:
    MULTIPLIER = 1e12
    NUM_TRIALS = 64
    TREE_DISTS = {}

    def __init__(self, tree, pastmlfile, obsdatafile):
        self.treefile = tree
        self.pastmlfile = pastmlfile
        self.obsdatafile = obsdatafile
        self.leaves = []
        self.node_map = {}
        self.simulation_result: pd.DataFrame = pd.DataFrame()
        self.trait_data: pd.DataFrame = pd.DataFrame()
        self.tree = None
        self._prepare_data()

    def _prepare_data(self):
        if isinstance(self.pastmlfile, pd.DataFrame):
            self.pastml = self.pastmlfile.copy()
        else:
            self.pastml = pd.read_csv(self.pastmlfile, index_col=0)
            # New pastml format: gene is the first column, so it becomes the index.
            # Reset it to a regular column to preserve legacy behavior.
            if self.pastml.index.name == 'gene':
                self.pastml = self.pastml.reset_index()

        if isinstance(self.obsdatafile, pd.DataFrame):
            self.obsdf = self.obsdatafile.copy()
        else:
            self.obsdf = pd.read_csv(self.obsdatafile, index_col=0)

        self._check_pastml_data()
        self._process_obs_data()

    def _process_obs_data(self):
        self.pastml = self.pastml.set_index('gene').loc[self.obsdf.columns].reset_index(names='gene')
        self.mapping = dict(zip(self.pastml.index.astype(str), self.pastml["gene"]))
        self.mappingr = dict(zip(self.pastml["gene"], self.pastml.index.astype(str)))
        self.obsdf[self.obsdf > 0.5] = 1
        self.obsdf.fillna(0, inplace=True)
        self.obsdf = self.obsdf.astype(int)
        self.obsdf.index = self.obsdf.index.astype(str)
        self.obsdf.rename(columns=self.mappingr, inplace=True)

    def _check_pastml_data(self):
        assert "gene" in self.pastml, "pastml file should have label `gene`"
        assert "gains" in self.pastml, "pastml file should have label `gains`"
        assert "losses" in self.pastml, "pastml file should have label `losses`"
        assert "dist" in self.pastml, "pastml file should have label `dist`"
        assert "loss_dist" in self.pastml, "pastml file should have label `loss_dist`"

    def initialize_simulation_parameters(self, pair_statistic=None, prevalence_threshold=0.05,
                                         collapse_theshold=0.001, single_trait=False,
                                         vars=None, targets=None, kde=False):
        """
        Initializes simulation parameters from pastml file and sets the pair_statistic method.
        Must be run before each simulation.

        :param pair_statistic: a function that takes two arrays and outputs a score (float)
        :param collapse_theshold: note: typo preserved from original API
        """
        if not self.tree:
            self.tree = Tree(self.treefile, 1)

        def check_internal_node_names(tree):
            internal_names = set()
            for node in tree.traverse():
                if not node.is_leaf():
                    if node.name in internal_names:
                        return False
                    internal_names.add(node.name)
            return True

        if not check_internal_node_names(self.tree):
            for idx, node in enumerate(self.tree.iter_descendants("levelorder")):
                if not node.is_leaf():
                    node.name = f"internal_{idx}"

        self.pair_statistic = pair_statistic or PairStatistics._vectorized_pair_statistic
        self.gains = np.array(self.pastml['gains'])
        self.losses = np.array(self.pastml['losses'])
        self.dists = np.array(self.pastml['dist'])
        self.loss_dists = np.array(self.pastml['loss_dist'])
        self.gain_subsize = np.array(self.pastml['gain_subsize'])
        self.loss_subsize = np.array(self.pastml['loss_subsize'])
        self.root_states = np.array(self.pastml['root_state'])
        self.dists[self.dists == np.inf] = 0
        self.loss_dists[self.loss_dists == np.inf] = 0
        self.kde = kde

        self.obsdf_modified = self._collapse_tree_tips(collapse_theshold)
        if vars and targets:
            self.set_pairs(vars, targets, by='name')
        else:
            self.pairs, self.obspairs = self._get_pair_data(
                self.obsdf_modified, self.obsdf_modified, prevalence_threshold, single_trait
            )

    def set_pairs(self, vars, targets, by: Literal['number', 'name'] = 'name'):
        if by == 'name':
            obsdf = self.get_obs()
            self.pairs, self.obspairs = self._get_pair_data(
                obsdf[vars].rename(columns=self.mappingr),
                obsdf[targets].rename(columns=self.mappingr)
            )
        else:
            obsdf = self.obsdf_modified
            self.pairs, self.obspairs = self._get_pair_data(
                obsdf[[str(i) for i in vars]],
                obsdf[[str(i) for i in targets]]
            )

    def _collapse_tree_tips(self, threshold):
        if threshold == 0:
            treeleaves = set(self.tree.get_leaf_names())
            self.tree.prune([i for i in self.obsdf.index if i in treeleaves],
                            preserve_branch_length=True)
            return self.obsdf.copy()

        threshold = self.tree.get_distance(self.tree, self.tree.get_farthest_leaf()[0]) * threshold
        obsdf = self.obsdf.copy()
        self.tree.prune([i for i in self.obsdf.index if i in set(self.tree.get_leaf_names())],
                        preserve_branch_length=True)
        node_queue = set(self.tree.get_leaves())
        to_prune = set()
        while node_queue:
            current_node: TreeNode = node_queue.pop()
            sibling: TreeNode = current_node.get_sisters()[0]
            if not sibling.is_leaf():
                to_prune.add(current_node)
                continue
            distance = current_node.dist + sibling.dist
            if distance < threshold:
                obsdf.loc[current_node.up.name] = obsdf.loc[current_node.name]
                node_queue.add(current_node.up)
            else:
                to_prune.add(current_node)
                to_prune.add(sibling)
            if sibling in node_queue:
                node_queue.remove(sibling)

        self.tree.prune(to_prune, preserve_branch_length=True)
        obsdf[obsdf > 1] = 1
        return obsdf.loc(axis=0)[tuple(map(lambda x: x.name, to_prune))]

    def _get_pair_data(self, vars: pd.DataFrame, targets: pd.DataFrame,
                       prevalence_threshold: float = 0.00, single_trait: bool = False,
                       batch_size=1000) -> Tuple[np.ndarray, np.ndarray]:
        vars_np = vars.to_numpy()
        targets_np = targets.to_numpy()
        var_cols = np.array(vars.columns)
        target_cols = np.array(targets.columns)

        valid_vars_mask = (vars_np.sum(axis=0) >= prevalence_threshold * vars_np.shape[0])
        valid_targets_mask = (targets_np.sum(axis=0) >= prevalence_threshold * targets_np.shape[0])

        valid_vars = var_cols[valid_vars_mask]
        valid_targets = target_cols[valid_targets_mask]

        if valid_vars.size == 0 or valid_targets.size == 0:
            return np.array([]), np.array([])

        if single_trait:
            pairs = np.array(np.meshgrid(valid_vars, valid_targets)).T.reshape(-1, 2)[:valid_vars.size]
        elif vars.equals(targets):
            pairs = np.array(list(combinations(valid_vars, 2)))
        else:
            pairs = np.array(np.meshgrid(valid_vars, valid_targets)).T.reshape(-1, 2)
            pairs_sorted = np.sort(pairs, axis=1)
            pairs = np.unique(pairs_sorted, axis=0)

        all_stats = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            pair_vars = vars[batch_pairs[:, 0]].to_numpy()
            pair_targets = targets[batch_pairs[:, 1]].to_numpy()
            batch_stats = self.pair_statistic(pair_vars, pair_targets)
            all_stats.append(batch_stats)

        stats = np.concatenate(all_stats, axis=0)
        pairs = pairs.astype(int)
        return pairs, stats

    def _get_pair_data2(self, obsdf: pd.DataFrame, pairs: List[Tuple],
                        prevalence_threshold: float = 0.00,
                        batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        obs_np = obsdf.to_numpy()
        obs_cols = np.array(obsdf.columns)
        col_idx = {col: i for i, col in enumerate(obs_cols)}

        pair_indices = np.array([(col_idx[str(i)], col_idx[str(j)]) for i, j in pairs])

        prevalence_counts = (obs_np != 0).sum(axis=0)
        valid_mask = prevalence_counts >= (prevalence_threshold * obs_np.shape[0])

        valid_pairs_mask = valid_mask[pair_indices[:, 0]] & valid_mask[pair_indices[:, 1]]
        valid_pairs = pair_indices[valid_pairs_mask]

        if valid_pairs.size == 0:
            return np.array([]), np.array([])

        all_stats = []
        for i in range(0, len(valid_pairs), batch_size):
            batch_pairs = valid_pairs[i:i + batch_size]
            pair_vars = obs_np[:, batch_pairs[:, 0]]
            pair_targets = obs_np[:, batch_pairs[:, 1]]
            batch_stats = self.pair_statistic(pair_vars, pair_targets)
            all_stats.append(batch_stats)

        stats = np.concatenate(all_stats, axis=0)
        return valid_pairs, stats

    def get_obs(self):
        try:
            return self.obsdf_modified.rename(columns=self.mapping)
        except Exception:
            return self.obsdf.rename(columns=self.mapping)

    def run_simulation(self, simulation_function=None, parallel=True, bit=True, norm=True):
        if not simulation_function:
            simulation_function = simulate_distnorm
        self.parallel = parallel
        if not bit:
            self.simulation_result, self.trait_data, self.sim_trees = simulation_function(self)
            self._post_process_simulation_results()
        else:
            self.simulation_result = simulate_glrates_bit(self) if not norm else simulate_glrates_bit_norm(self)
            self.simulation_result['sys1'] = [self.mapping[str(i)] for i in self.simulation_result['first']]
            self.simulation_result['sys2'] = [self.mapping[str(i)] for i in self.simulation_result['second']]

    def _post_process_simulation_results(self):
        self.simulation_result['sys1'] = [self.mapping[str(i)] for i in self.simulation_result['first']]
        self.simulation_result['sys2'] = [self.mapping[str(i)] for i in self.simulation_result['second']]
        self.trait_data['gene'] = [self.mapping[str(i)] for i in self.trait_data['trait']]
        self.trait_data['obs'] = [self.obsdf_modified[str(i)].replace(0, np.nan).count()
                                  for i in self.trait_data['trait']]
        self.trait_data['z_score'] = [None if k == 0 else (i - j) / k
                                      for i, j, k in zip(self.trait_data['obs'],
                                                          self.trait_data['mean'],
                                                          self.trait_data['std'])]
        self.trait_data['robust_z_score'] = [None if k == 0 else (i - j) / k
                                             for i, j, k in zip(self.trait_data['obs'],
                                                                 self.trait_data['median'],
                                                                 self.trait_data['iqr'])]

    def get_simulation_result(self):
        if self.simulation_result.empty:
            raise ValueError("Simulation not yet run.")
        return self.simulation_result

    def get_trait_data(self):
        if self.trait_data.empty:
            raise ValueError("Simulation was either not yet completed or run with bit-optimized computation.")
        return self.trait_data

    def set_trials(self, num_trials: int) -> None:
        self.NUM_TRIALS = num_trials

    def get_top_results(self, correction: Union[bool, str] = False, prevalence_range=[0, 1],
                        top=15, direction: Literal[-1, 0, 1] = 0,
                        by: Literal['p-value', 'effect size'] = 'effect size', alpha=0.05):
        res = self._filter_res(correction, prevalence_range, alpha=alpha)

        if direction:
            res = res[res['direction'] == direction]

        prev = self.obsdf_modified.mean()
        res['prevalence_sys1'] = res['first'].astype(str).map(prev)
        res['prevalence_sys2'] = res['second'].astype(str).map(prev)
        res['effect size'] = abs(res['effect size'])

        return res[['sys1', 'sys2', 'direction', 'p-value', 'effect size',
                    'prevalence_sys1', 'prevalence_sys2']].sort_values(
            by=by, ascending=(by == 'p-value')).head(top)

    def _filter_res(self, correction, prevalence_range, alpha=0.05):
        res = self.simulation_result[self.simulation_result['first'] != self.simulation_result['second']]

        if prevalence_range != [0, 1]:
            prev = self.obsdf_modified.sum() / self.obsdf_modified.count()
            to_keep = prev[(prev >= prevalence_range[0]) & (prev <= prevalence_range[1])]
            res = res[res['first'].isin(to_keep.index.astype(int))]
            res = res[res['second'].isin(to_keep.index.astype(int))]

        if correction:
            method = correction if type(correction) == str else 'fdr_bh'
            bhc_s = sm.multipletests(res['p-value_syn'], alpha=alpha, method=method)
            bhc_a = sm.multipletests(res['p-value_ant'], alpha=alpha, method=method)
            res.loc[:, 'p-value_syn'] = bhc_s[1]
            res.loc[:, 'p-value_ant'] = bhc_a[1]
            res.loc[:, 'p-value'] = np.minimum(res.loc[:, 'p-value_syn'], res.loc[:, 'p-value_ant'])

        return res

    def fisher(self, traits: list) -> None:
        assert len(traits) == 2, "must be given two traits"
        table = [[0, 0], [0, 0]]
        d = self.obsdf_modified.rename(columns=self.mapping)
        for a1, a2 in zip(d[traits[0]], d[traits[1]]):
            table[a1][a2] += 1
        odds_ratio, p_value = fisher_exact(table)
        print("Fischer's Exact Results:")
        print(f"Odds Ratio: {odds_ratio}")
        print(f"P-value: {p_value}")

    def plot_effect_size(self, correction: Union[bool, str] = False, prevalence_range=[0, 1]):
        if not _HAS_PLOTLY:
            raise ImportError("plotly is required for plot_effect_size()")
        x = self._filter_res(correction, prevalence_range)
        if np.any(x['p-value'] == 0):
            x.loc[x['p-value'] == 0, 'p-value'] = 0.1 / x.loc[x['p-value'] == 0, 'num_pair_trials']
        x = x.assign(
            log=lambda x: -np.log10(x['p-value']),
            pair=lambda x: list(zip(x['sys1'], x['sys2']))
        )
        fig = px.scatter(x, x='effect size', y='log', hover_data=['pair', 'direction'],
                         labels={'log': '-log p-value', 'effect size': 'Effect Size'})
        fig.update_layout(title='P-value vs Effect Size')
        fig.show()
