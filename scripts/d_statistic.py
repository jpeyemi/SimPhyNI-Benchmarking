"""
D-statistic (Fritz & Purvis 2010) for binary phylogenetic traits.

D measures the phylogenetic signal of a binary trait:
  D = (observed_diffs - brownian_mean) / (random_mean - brownian_mean)

  D < 0  : more clumped than Brownian (clade-driven)
  D ≈ 0  : Brownian motion level of clumping
  D ≈ 1  : phylogenetically random

Public API
----------
precompute_tree_structure(tree)
    Walk an ETE3 tree once and return index arrays for O(n) contrast computation.

simulate_bm_vectors(tree_structure, tree_ete3, n_permutations=999)
    Simulate Brownian motion on the tree once and return raw continuous leaf
    value matrix of shape (n_permutations, n_leaves).  Reused across all traits.

get_or_calibrate(tree_path, tree_ete3, n_permutations=999)
    Cached wrapper: runs precompute_tree_structure + simulate_bm_vectors once
    per unique tree path and returns (tree_structure, bm_leaf_vals).

get_null_distributions(tree_path, tree_structure, bm_leaf_vals, prevalence, n_permutations=999)
    Return (random_mean, brownian_mean) for a specific trait prevalence.
    Results are memoized by (tree_path, round(prevalence, 3)) so identical
    prevalences share one computation.

compute_d_statistic(tree_structure, tip_states, random_mean, brownian_mean)
    Compute D for a single binary tip-state vector in O(n).
"""

import numpy as np

# tree_path → (tree_structure, bm_leaf_vals)
_TREE_CACHE = {}

# (tree_path, rounded_prevalence) → (random_mean, brownian_mean)
_PREVALENCE_CACHE = {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_leaves_of(node):
    """Return all leaf nodes descended from *node* (inclusive if leaf)."""
    if node.is_leaf():
        return [node]
    return node.get_leaves()


def _contrast_sums(binary_matrix, node_left_indices, node_right_indices):
    """
    Compute per-row sum of absolute clade-mean differences.

    Parameters
    ----------
    binary_matrix      : np.ndarray, shape (n_rows, n_leaves)
    node_left_indices  : list[np.ndarray]
    node_right_indices : list[np.ndarray]

    Returns
    -------
    np.ndarray, shape (n_rows,)
    """
    totals = np.zeros(binary_matrix.shape[0])
    for L, R in zip(node_left_indices, node_right_indices):
        totals += np.abs(
            binary_matrix[:, L].mean(axis=1) - binary_matrix[:, R].mean(axis=1)
        )
    return totals


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def precompute_tree_structure(tree):
    """
    Walk an ETE3 tree once (postorder) and extract the index arrays needed for
    fast O(n) contrast-sum computation.

    Parameters
    ----------
    tree : ete3.Tree  (rooted; leaves must be named)

    Returns
    -------
    dict with keys
        leaf_names        : list[str]          — stable ordering of all leaves
        leaf_to_idx       : dict[str, int]
        node_left_indices : list[np.ndarray]   — one per internal node
        node_right_indices: list[np.ndarray]   — parallel to node_left_indices
    """
    leaf_names = [n.name for n in tree.get_leaves()]
    if len(leaf_names) != len(set(leaf_names)):
        from collections import Counter
        dupes = [name for name, cnt in Counter(leaf_names).items() if cnt > 1]
        raise ValueError(
            f"Tree has {len(dupes)} non-unique leaf name(s): {dupes[:5]}"
            f"{'...' if len(dupes) > 5 else ''}. "
            "Deduplicate leaf names before calling precompute_tree_structure()."
        )
    leaf_to_idx = {name: i for i, name in enumerate(leaf_names)}

    node_left_indices  = []
    node_right_indices = []

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            continue
        if len(node.children) < 2:
            # Degenerate node with a single child — skip (no contrast possible)
            continue

        # First child = "left"; everything else collapsed into "right"
        left_leaves = [leaf_to_idx[l.name] for l in _get_leaves_of(node.children[0])]
        right_leaves = []
        for child in node.children[1:]:
            right_leaves.extend(leaf_to_idx[l.name] for l in _get_leaves_of(child))

        node_left_indices.append(np.array(left_leaves,  dtype=np.int32))
        node_right_indices.append(np.array(right_leaves, dtype=np.int32))

    return {
        "leaf_names":         leaf_names,
        "leaf_to_idx":        leaf_to_idx,
        "node_left_indices":  node_left_indices,
        "node_right_indices": node_right_indices,
    }


def simulate_bm_vectors(tree_structure, tree_ete3, n_permutations=999):
    """
    Simulate Brownian motion on the tree and return raw continuous leaf values.

    This is the expensive per-tree step.  The returned matrix is shared across
    all traits; each trait thresholds it at its own prevalence in
    get_null_distributions().

    Parameters
    ----------
    tree_structure : dict    — output of precompute_tree_structure
    tree_ete3      : ete3.Tree
    n_permutations : int

    Returns
    -------
    bm_leaf_vals : np.ndarray, shape (n_permutations, n_leaves)
        Raw continuous BM tip values (not thresholded).
    """
    leaf_names = tree_structure["leaf_names"]
    n_leaves   = len(leaf_names)

    rng = np.random.default_rng(seed=42)

    nodes_pre   = list(tree_ete3.traverse("preorder"))
    node_to_idx = {id(n): i for i, n in enumerate(nodes_pre)}
    n_nodes     = len(nodes_pre)

    # (n_permutations, n_nodes); root stays at 0
    bm_vals = np.zeros((n_permutations, n_nodes))

    for node in tree_ete3.traverse("preorder"):
        if node.is_root():
            continue
        p_idx = node_to_idx[id(node.up)]
        n_idx = node_to_idx[id(node)]
        bl    = max(node.dist, 1e-12)
        bm_vals[:, n_idx] = bm_vals[:, p_idx] + rng.normal(
            0.0, np.sqrt(bl), size=n_permutations
        )

    # Extract leaf columns in the same order as leaf_names
    leaf_to_node_idx = {
        n.name: node_to_idx[id(n)]
        for n in nodes_pre if n.is_leaf()
    }
    leaf_node_cols = [leaf_to_node_idx[name] for name in leaf_names]
    return bm_vals[:, leaf_node_cols]   # (n_permutations, n_leaves)


def get_null_distributions(tree_path, tree_structure, bm_leaf_vals, prevalence,
                           n_permutations=999):
    """
    Return (random_mean, brownian_mean) conditioned on a trait's prevalence.

    Results are memoized by (tree_path, round(prevalence, 3)).

    Parameters
    ----------
    tree_path      : str           — cache key (file path string)
    tree_structure : dict          — output of precompute_tree_structure
    bm_leaf_vals   : np.ndarray    — shape (n_permutations, n_leaves), from simulate_bm_vectors
    prevalence     : float         — observed prevalence of the trait (tip_states.mean())
    n_permutations : int

    Returns
    -------
    (random_mean, brownian_mean) : (float, float)
    """
    rounded = round(prevalence, 3)
    key = (tree_path, rounded)
    if key in _PREVALENCE_CACHE:
        return _PREVALENCE_CACHE[key]

    node_left_indices  = tree_structure["node_left_indices"]
    node_right_indices = tree_structure["node_right_indices"]
    n_leaves           = bm_leaf_vals.shape[1]

    # ── random_mean: permutation null at trait's prevalence ───────────────────
    n_ones = max(1, round(rounded * n_leaves))
    base   = np.zeros(n_leaves)
    base[:n_ones] = 1.0

    # Use a deterministic seed derived from the key so results are reproducible
    seed = hash(key) & 0xFFFFFFFF
    rng  = np.random.default_rng(seed=seed)
    perms = np.vstack([rng.permutation(base) for _ in range(n_permutations)])

    random_mean = float(_contrast_sums(perms, node_left_indices, node_right_indices).mean())

    # ── brownian_mean: threshold BM matrix at trait's prevalence ─────────────
    # For each BM replicate (row), find the value at the (1 - rounded) quantile
    # so that approximately `rounded` fraction of leaves are "present".
    threshold  = np.quantile(bm_leaf_vals, 1.0 - rounded, axis=1, keepdims=True)
    bm_binary  = (bm_leaf_vals >= threshold).astype(float)

    brownian_mean = float(
        _contrast_sums(bm_binary, node_left_indices, node_right_indices).mean()
    )

    _PREVALENCE_CACHE[key] = (random_mean, brownian_mean)
    return random_mean, brownian_mean


def compute_d_statistic(tree_structure, tip_states, random_mean, brownian_mean):
    """
    Compute D for a single binary tip-state vector.

    Parameters
    ----------
    tree_structure   : dict   — output of precompute_tree_structure
    tip_states       : np.ndarray, shape (n_leaves,), dtype float64
                       Values 0.0 / 1.0, ordered by tree_structure["leaf_names"].
    random_mean      : float  — from get_null_distributions
    brownian_mean    : float  — from get_null_distributions

    Returns
    -------
    D : float, or float('nan') for degenerate inputs
    """
    node_left_indices  = tree_structure["node_left_indices"]
    node_right_indices = tree_structure["node_right_indices"]

    # Guard: degenerate prevalence
    total = tip_states.sum()
    n     = len(tip_states)
    if total == 0 or total == n or total < 2:
        return float("nan")

    # Guard: uninformative calibration (e.g. perfect star tree)
    if abs(random_mean - brownian_mean) < 1e-12:
        return float("nan")

    observed = 0.0
    for L, R in zip(node_left_indices, node_right_indices):
        observed += abs(tip_states[L].mean() - tip_states[R].mean())

    return (observed - brownian_mean) / (random_mean - brownian_mean)


def get_or_calibrate(tree_path, tree_ete3, n_permutations=999):
    """
    Return cached (tree_structure, bm_leaf_vals) for this tree,
    running precomputation on first call.

    Parameters
    ----------
    tree_path      : str        — used as cache key (the file path string)
    tree_ete3      : ete3.Tree
    n_permutations : int

    Returns
    -------
    (tree_structure, bm_leaf_vals)
    """
    if tree_path not in _TREE_CACHE:
        ts           = precompute_tree_structure(tree_ete3)
        bm_leaf_vals = simulate_bm_vectors(ts, tree_ete3, n_permutations)
        _TREE_CACHE[tree_path] = (ts, bm_leaf_vals)
    return _TREE_CACHE[tree_path]
