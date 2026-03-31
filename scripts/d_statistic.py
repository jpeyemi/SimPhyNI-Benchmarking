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

calibrate_d_nulls(tree_structure, tree_ete3, n_permutations=999)
    Estimate random_mean and brownian_mean for a given tree topology.
    These are tree-level constants reused across all trait pairs.

compute_d_statistic(tree_structure, tip_states, random_mean, brownian_mean)
    Compute D for a single binary tip-state vector in O(n).

get_or_calibrate(tree_path, tree_ete3, n_permutations=999)
    Cached wrapper: calibrates once per unique tree path and returns
    (tree_structure, random_mean, brownian_mean).
"""

import numpy as np

# Module-level cache: tree_path → (tree_structure, random_mean, brownian_mean)
_CACHE = {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_leaves_of(node):
    """Return all leaf nodes descended from *node* (inclusive if leaf)."""
    if node.is_leaf():
        return [node]
    return node.get_leaves()


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
    leaf_names  = [n.name for n in tree.get_leaves()]
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


def calibrate_d_nulls(tree_structure, tree_ete3, n_permutations=999):
    """
    Estimate random_mean and brownian_mean for this tree topology.

    Both values are scalars that depend only on the tree, not on any trait
    vector — compute once and reuse across all pairs on the same tree.

    Parameters
    ----------
    tree_structure : dict    — output of precompute_tree_structure
    tree_ete3      : ete3.Tree
    n_permutations : int

    Returns
    -------
    (random_mean, brownian_mean) : (float, float)
    """
    leaf_names         = tree_structure["leaf_names"]
    node_left_indices  = tree_structure["node_left_indices"]
    node_right_indices = tree_structure["node_right_indices"]
    n_leaves           = len(leaf_names)

    rng = np.random.default_rng(seed=42)

    # ── random_mean: tip-label permutation ───────────────────────────────────
    # Use a balanced 50/50 binary vector as base; permute it n_permutations times.
    # D's random_mean is a property of tree topology alone.
    base = np.zeros(n_leaves)
    base[: n_leaves // 2] = 1.0

    # Build (n_permutations, n_leaves) matrix row-by-row
    perms = np.vstack([rng.permutation(base) for _ in range(n_permutations)])

    contrast_sums = np.zeros(n_permutations)
    for L, R in zip(node_left_indices, node_right_indices):
        contrast_sums += np.abs(perms[:, L].mean(axis=1) - perms[:, R].mean(axis=1))

    random_mean = float(contrast_sums.mean())

    # ── brownian_mean: BM simulation → binary ────────────────────────────────
    # Propagate Gaussian increments (variance = branch_length) from root to tips
    # in preorder, then threshold at row-wise median to get binary states.
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
    leaf_vals = bm_vals[:, leaf_node_cols]   # (n_permutations, n_leaves)

    # Threshold at row-wise median → binary
    medians   = np.median(leaf_vals, axis=1, keepdims=True)
    bm_binary = (leaf_vals >= medians).astype(float)

    contrast_sums_bm = np.zeros(n_permutations)
    for L, R in zip(node_left_indices, node_right_indices):
        contrast_sums_bm += np.abs(
            bm_binary[:, L].mean(axis=1) - bm_binary[:, R].mean(axis=1)
        )

    brownian_mean = float(contrast_sums_bm.mean())

    return random_mean, brownian_mean


def compute_d_statistic(tree_structure, tip_states, random_mean, brownian_mean):
    """
    Compute D for a single binary tip-state vector.

    Parameters
    ----------
    tree_structure   : dict   — output of precompute_tree_structure
    tip_states       : np.ndarray, shape (n_leaves,), dtype float64
                       Values 0.0 / 1.0, ordered by tree_structure["leaf_names"].
    random_mean      : float  — from calibrate_d_nulls
    brownian_mean    : float  — from calibrate_d_nulls

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
    Return cached (tree_structure, random_mean, brownian_mean) for this tree,
    running calibration on first call.

    Parameters
    ----------
    tree_path    : str        — used as cache key (the file path string)
    tree_ete3    : ete3.Tree
    n_permutations : int

    Returns
    -------
    (tree_structure, random_mean, brownian_mean)
    """
    if tree_path not in _CACHE:
        ts = precompute_tree_structure(tree_ete3)
        rm, bm = calibrate_d_nulls(ts, tree_ete3, n_permutations)
        _CACHE[tree_path] = (ts, rm, bm)
    return _CACHE[tree_path]
