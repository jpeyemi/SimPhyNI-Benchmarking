"""
sim_wrappers.py — Simulation wrapper functions for updated parameter traversal.

Provides four simulation functions that produce lineage arrays using the current
SimPhyNI bit-packed backend, plus a generic result compilation function
(_compres_custom) that supports any pair statistic (not just LOR).

Each ``get_lineages_*`` function runs the tree simulation once and returns
``(lineages, mappingr, to_index_fn)`` so the caller can compile results for
multiple statistics without repeating the simulation.

``compile_results`` uses ``_compres_custom`` for p-value estimation:
  - KDE (gaussian_kde): for LOR and log_add_ratio (continuous, unbounded)
  - Empirical counting:  for all other statistics

No SimPhyNI source files are modified by this module.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "SimPhyNI"))

from simphyni.Simulation.simulation import (
    sim_bit,
    build_sim_params,
    unpack_trait_params,
    _chunk_info,
)


# ---------------------------------------------------------------------------
# Index mapping helpers
# ---------------------------------------------------------------------------

def _build_index_maps(trait_params):
    """
    Build bidirectional maps between trait_params positional indices and gene names.

    ``build_sim_params`` returns integer-indexed rows with a 'gene' column, so
    gene names come from trait_params['gene'], not the index.

    Returns
    -------
    mappingr : dict  int -> gene_name
    to_index_fn : callable  gene_name_array -> int_array
    """
    if 'gene' in trait_params.columns:
        gene_names = trait_params['gene'].tolist()
    else:
        gene_names = trait_params.index.tolist()
    mappingr = dict(enumerate(gene_names))
    mapping  = {name: i for i, name in enumerate(gene_names)}
    # str() coercion handles np.str_ keys from numpy string arrays
    to_index_fn = lambda pairs: np.vectorize(lambda k: mapping[str(k)])(pairs)
    return mappingr, to_index_fn


# ---------------------------------------------------------------------------
# Custom result compilation — supports any pair statistic
# ---------------------------------------------------------------------------

def _compres_custom(lineages, pairs, obspairs, stat_fn, use_kde=True,
                    trials=64, batch_size=100):
    """
    Compile p-values for any pair statistic against its own simulated null
    distribution.  Replaces compres() (which hardcodes LOR) for this pipeline.

    For each pair the null distribution is built by rotating one trait's trial
    vector relative to the other (trials² combinations) and evaluating
    ``stat_fn`` at each combination, exactly as the legacy compile_results_KDE
    approach.

    Parameters
    ----------
    lineages  : ndarray, shape (n_tips, n_traits, n_chunks), dtype uint64
                Output of sim_bit or _sim_bit_ctmp.
    pairs     : ndarray, shape (n_pairs, 2), dtype int
                Column indices into the trait axis of ``lineages``.
    obspairs  : ndarray, shape (n_pairs,), dtype float
                Observed statistic values — must be computed with the same
                ``stat_fn``.
    stat_fn   : callable(tp, tq) -> ndarray, shape (n_pairs,)
                tp and tq are boolean arrays of shape (n_tips, n_pairs).
    use_kde   : bool
                If True use gaussian_kde for p-values (KDE of simulated null).
                If False use empirical counting (fraction of null >= / <= obs).
    trials    : int  — number of bit-packed trials (must match lineages).
    batch_size: int  — pairs processed together per inner loop.

    Returns
    -------
    pd.DataFrame with columns: first, second, p-value, direction, effect size.
    ``first`` and ``second`` hold the integer pair indices (not yet gene names).
    """
    n_tips, n_traits, n_chunks = lineages.shape
    n_pairs = len(pairs)

    # Unpack each unique trait's bit array once
    unique_traits = np.unique(pairs)
    trait_bool = {}
    for t in unique_traits:
        packed = lineages[:, t, :]                                     # (n_tips, n_chunks) uint64
        raw = np.unpackbits(packed.view(np.uint8), axis=-1,
                            bitorder="little")                          # (n_tips, n_chunks*64)
        trait_bool[t] = raw[:, :trials].astype(bool)                   # (n_tips, trials)

    results = []
    for start in range(0, n_pairs, batch_size):
        bp = pairs[start: start + batch_size]                          # (nb, 2)
        bo = obspairs[start: start + batch_size]                       # (nb,)
        nb = len(bp)

        # (n_tips, nb, trials)
        tp_batch = np.stack([trait_bool[p] for p, _ in bp], axis=1)
        tq_batch = np.stack([trait_bool[q] for _, q in bp], axis=1)

        null_dist = np.empty((nb, trials * trials), dtype=np.float64)

        for k in range(trials):
            tq_rolled = np.roll(tq_batch, k, axis=-1)                  # rotate trial axis
            for t in range(trials):
                tp_t = tp_batch[:, :, t]                                # (n_tips, nb) bool
                tq_t = tq_rolled[:, :, t]
                vals = stat_fn(tp_t, tq_t)                              # (nb,) float
                null_dist[:, k * trials + t] = vals

        # Replace NaN (e.g. zero-variance z-stat) with null median
        null_dist = np.where(np.isfinite(null_dist), null_dist, 0.0)

        for i in range(nb):
            obs_i = bo[i]
            null_i = null_dist[i]

            if use_kde:
                jitter = np.random.normal(0, 1e-12, size=null_i.shape)
                kde = gaussian_kde(null_i + jitter, bw_method="silverman")
                kde_neg = gaussian_kde(-(null_i + jitter), bw_method="silverman")
                pval_ant = kde.integrate_box_1d(-np.inf, obs_i)
                pval_syn = kde_neg.integrate_box_1d(-np.inf, -obs_i)
            else:
                n_tot = len(null_i)
                pval_ant = float(np.sum(null_i <= obs_i)) / n_tot
                pval_syn = float(np.sum(null_i >= obs_i)) / n_tot

            pval = min(pval_ant, pval_syn)
            direction = -1 if pval_ant < pval_syn else 1
            med = np.median(null_i)
            iqr = float(np.percentile(null_i, 75) - np.percentile(null_i, 25))
            effect = (obs_i - med) / max(iqr * 1.349, 1.0)

            results.append({
                "first":       int(bp[i, 0]),
                "second":      int(bp[i, 1]),
                "p-value":     pval,
                "direction":   direction,
                "effect size": effect,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CTMP bit-packed simulation (closed-form transition probabilities)
# ---------------------------------------------------------------------------

def _sim_bit_ctmp(tree, trait_params, trials=64):
    """
    Bit-packed simulation using the exact closed-form 2-state CTMP transition
    probabilities instead of Gillespie stepping.

    For a branch of length t with gain rate λ and loss rate μ:
        q         = λ + μ   (total exit rate; only counts rates whose distance
                              threshold has been crossed)
        p(0 → 1)  = (λ/q) * (1 − exp(−q·t))
        p(1 → 0)  = (μ/q) * (1 − exp(−q·t))

    This correctly marginalises over all possible multi-flip paths within the
    branch (e.g. 0→1→0), unlike the Poisson approximation in sim_bit which
    applies gain and loss rates independently.  The output format is identical
    to sim_bit: shape (n_tips, n_traits, n_chunks) uint64.

    Parameters
    ----------
    tree         : ETE3 Tree with labelled internal nodes.
    trait_params : pd.DataFrame from build_sim_params() — canonical columns.
    trials       : int, number of bit-packed trials (must be >= 1).
    """
    gains, losses, dists, loss_dists, gain_subsize, loss_subsize, root_values = \
        unpack_trait_params(trait_params)

    valid_gains  = gain_subsize > 0
    valid_losses = loss_subsize > 0

    gain_rates = np.zeros(len(gains), dtype=float)
    loss_rates = np.zeros(len(losses), dtype=float)
    gain_rates[valid_gains]  = gains[valid_gains]  / gain_subsize[valid_gains]
    loss_rates[valid_losses] = losses[valid_losses] / loss_subsize[valid_losses]

    node_map = {node: idx for idx, node in enumerate(tree.traverse())}
    num_nodes  = len(node_map)
    num_traits = len(gains)
    bits       = 64
    nptype     = np.uint64
    n_chunks, last_chunk_bits = _chunk_info(trials)

    FULL_MASK = nptype(18446744073709551615)          # all 64 bits
    last_mask = FULL_MASK if last_chunk_bits == 64 \
        else (nptype(1) << nptype(last_chunk_bits)) - nptype(1)

    sim = np.zeros((num_nodes, num_traits, n_chunks), dtype=nptype)

    # Cumulative distances from root
    node_dists = {}
    node_dists[tree] = tree.dist or 0.0
    for node in tree.traverse():
        if node in node_dists:
            continue
        node_dists[node] = node_dists[node.up] + node.dist

    print("Simulating Trees (CTMP)...")
    for node in tree.traverse():
        node_idx = node_map[node]

        if node.up is None:
            # Root: set all bits for traits present at root
            root = root_values > 0
            for c in range(n_chunks):
                chunk_val = FULL_MASK if c < n_chunks - 1 else last_mask
                sim[node_idx, root, c] = chunk_val
            continue

        parent = sim[node_map[node.up], :, :]        # (n_traits, n_chunks)
        node_total_dist = node_dists[node]
        t = node.dist

        # Distance-threshold eligibility (same gating as sim_bit)
        eligible_gain  = node_total_dist >= dists
        eligible_loss  = node_total_dist >= loss_dists

        # Total exit rate per trait (zero if neither threshold met)
        q = gain_rates * eligible_gain.astype(float) + \
            loss_rates * eligible_loss.astype(float)

        # Closed-form CTMP transition probabilities.
        # Use np.divide with where= to avoid divide-by-zero warnings when q=0.
        factor    = np.where(q > 0, 1.0 - np.exp(-q * t), 0.0)
        safe_q    = np.where(q > 0, q, 1.0)              # dummy denominator for q=0 rows
        p_0_to_1  = np.where(q > 0, (gain_rates * eligible_gain) / safe_q * factor, 0.0)
        p_1_to_0  = np.where(q > 0, (loss_rates * eligible_loss) / safe_q * factor, 0.0)

        eligible = eligible_gain | eligible_loss
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            sim[node_idx, :, :] = parent
            continue

        p01 = p_0_to_1[eligible]                     # (n_eligible,)
        p10 = p_1_to_0[eligible]

        flip_gains  = np.zeros((num_traits, n_chunks), dtype=nptype)
        flip_losses = np.zeros((num_traits, n_chunks), dtype=nptype)

        for c in range(n_chunks):
            chunk_trials = bits if c < n_chunks - 1 else last_chunk_bits

            # Draw Bernoulli(p01) for gain-eligible traits
            g_samp = (np.random.uniform(size=(n_eligible, chunk_trials))
                      < p01[:, np.newaxis]).astype(np.uint8)
            # Draw Bernoulli(p10) for loss-eligible traits
            l_samp = (np.random.uniform(size=(n_eligible, chunk_trials))
                      < p10[:, np.newaxis]).astype(np.uint8)

            if chunk_trials < bits:
                g_samp = np.pad(g_samp, ((0, 0), (0, bits - chunk_trials)))
                l_samp = np.pad(l_samp, ((0, 0), (0, bits - chunk_trials)))

            flip_gains[eligible, c] = (
                np.packbits(g_samp, axis=-1, bitorder="little")
                .view(nptype).flatten()
            )
            flip_losses[eligible, c] = (
                np.packbits(l_samp, axis=-1, bitorder="little")
                .view(nptype).flatten()
            )

        # Apply conditioned on parent state:
        # gain only where parent bit = 0; loss only where parent bit = 1
        gain_events = flip_gains & ~parent
        loss_events = flip_losses & parent

        sim[node_idx, :, :] = (parent | gain_events) & ~loss_events

    print("Completed Tree Simulation (CTMP) Successfully")
    lineages = sim[[node_map[node] for node in tree], :, :]
    return lineages


# ---------------------------------------------------------------------------
# Four get_lineages_* functions — one per simulation method
# ---------------------------------------------------------------------------

def get_lineages_simulate(tree, trait_params, trials=64):
    """
    Standard bit-packed Poisson simulation (SimPhyNI default).

    Parameters
    ----------
    tree         : ETE3 Tree with labelled internal nodes.
    trait_params : pd.DataFrame from build_sim_params().
    trials       : int, number of independent trials.

    Returns
    -------
    lineages     : ndarray (n_tips, n_traits, n_chunks) uint64
    mappingr     : dict  int -> gene_name
    to_index_fn  : callable  gene_name_array -> int_array
    """
    lineages = sim_bit(tree, trait_params, trials=trials)
    mappingr, to_index_fn = _build_index_maps(trait_params)
    return lineages, mappingr, to_index_fn


def get_lineages_nodist(tree, acr_df, counting="JOINT", subsize="ORIGINAL",
                        trials=64):
    """
    Simulation with the distance-threshold gate disabled (dist = loss_dist = 0).

    Accepts the raw ACR DataFrame (not pre-built trait_params) because
    ``no_threshold=True`` must be applied at build_sim_params time.

    Parameters
    ----------
    tree     : ETE3 Tree with labelled internal nodes.
    acr_df   : pd.DataFrame — raw ACR CSV loaded by the caller.
    counting : str, 'JOINT' or 'FLOW'.
    subsize  : str, 'ORIGINAL', 'NO_FILTER', or 'THRESH'.
    trials   : int, number of independent trials.

    Returns
    -------
    lineages, mappingr, to_index_fn
    """
    trait_params = build_sim_params(acr_df, counting=counting, subsize=subsize,
                                    no_threshold=True)
    lineages = sim_bit(tree, trait_params, trials=trials)
    mappingr, to_index_fn = _build_index_maps(trait_params)
    return lineages, mappingr, to_index_fn


def get_lineages_ctmp(tree, trait_params, trials=64):
    """
    Closed-form CTMP simulation (exact 2-state Markov transition probabilities,
    bit-packed).  Same speed as the standard Poisson simulation but correctly
    handles multi-flip paths within a branch.

    Parameters
    ----------
    tree         : ETE3 Tree with labelled internal nodes.
    trait_params : pd.DataFrame from build_sim_params().
    trials       : int, number of independent trials.

    Returns
    -------
    lineages, mappingr, to_index_fn
    """
    lineages = _sim_bit_ctmp(tree, trait_params, trials=trials)
    mappingr, to_index_fn = _build_index_maps(trait_params)
    return lineages, mappingr, to_index_fn


def get_lineages_norm(tree, trait_params, trials=64):
    """
    Simulation with all branch lengths rescaled to 1.0 (uniform / topological).

    Original branch lengths are saved and restored in a ``finally`` block so
    the tree object is always left in its original state even if an exception
    occurs.

    Parameters
    ----------
    tree         : ETE3 Tree with labelled internal nodes.
    trait_params : pd.DataFrame from build_sim_params().
    trials       : int, number of independent trials.

    Returns
    -------
    lineages, mappingr, to_index_fn
    """
    original_dists = {node: node.dist for node in tree.traverse()}
    try:
        for node in tree.traverse():
            node.dist = 1.0
        lineages = sim_bit(tree, trait_params, trials=trials)
    finally:
        for node, dist in original_dists.items():
            node.dist = dist
    mappingr, to_index_fn = _build_index_maps(trait_params)
    return lineages, mappingr, to_index_fn


# ---------------------------------------------------------------------------
# Compile results — orchestrator
# ---------------------------------------------------------------------------

def compile_results(lineages, pairs, obspairs, stat_fn, use_kde,
                    mappingr, to_index_fn, trials=64, batch_size=100):
    """
    Compile simulation results for a given pair statistic.

    Parameters
    ----------
    lineages    : ndarray (n_tips, n_traits, n_chunks) uint64
    pairs       : array-like of (gene_name, gene_name) tuples — same format as
                  returned by TreeSimulator._get_pair_data2.
    obspairs    : ndarray (n_pairs,) — observed statistic values from stat_fn.
    stat_fn     : callable — the pair statistic used for both obspairs and null.
    use_kde     : bool — KDE p-values (True) or empirical counting (False).
    mappingr    : dict int -> gene_name (from get_lineages_*).
    to_index_fn : callable gene_name_array -> int_array (from get_lineages_*).
    trials      : int — must match the trials used when producing lineages.
    batch_size  : int — pairs per inner batch in _compres_custom.

    Returns
    -------
    pd.DataFrame with columns: first, second, p-value, direction, effect size.
    ``first`` and ``second`` contain gene name strings.
    """
    pairs_index = to_index_fn(np.array(pairs))                        # (n_pairs, 2) int
    res = _compres_custom(lineages, pairs_index, obspairs, stat_fn,
                          use_kde=use_kde, trials=trials,
                          batch_size=batch_size)
    res["first"]  = res["first"].map(mappingr)
    res["second"] = res["second"].map(mappingr)
    return res
