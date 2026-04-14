import numpy as np
from ete3 import Tree
import pandas as pd
import pickle
from scipy.stats import gaussian_kde
from scipy.special import expit

# Module-level caches
_KDE_CACHE = {}
_ACR_CACHE = {}
_ACR_HYBRID_CACHE = {}
_ACR_MARGINAL_CACHE = {}


def _load_acr(path='pastmlout_marginal.csv'):
    """Load ACR CSV and compute direct gain/loss rates.

    gain_rate = gains / gain_subsize
    loss_rate = losses / loss_subsize

    Rows where either subsize is zero or either rate is non-positive are
    excluded.  Cached after first load.
    """
    if path not in _ACR_CACHE:
        df = pd.read_csv(path)
        df = df[(df['gain_subsize'] > 0) & (df['loss_subsize'] > 0)].copy()
        df['g'] = df['gains'] / df['gain_subsize']
        df['l'] = df['losses'] / df['loss_subsize']
        df = df[(df['g'] > 0) & (df['l'] > 0)].reset_index(drop=True)
        _ACR_CACHE[path] = df
    return _ACR_CACHE[path]


def _load_acr_hybrid_m(path='pastmlout_marginal.csv'):
    """Load ACR data for the hybrid_m rate model.

    π  = g_marg / (g_marg + l_marg)  where
         g_marg = gains_flow / gain_subsize_marginal  (marginal rates → correct low-prevalence π)
         l_marg = losses_flow / loss_subsize_marginal
    λ  = (gains + losses) / ecoli_total_bl  (integer counts → preserves heterogeneity,
         includes genes with gains=0 or losses=0 via naturally low λ)

    Genes with gains + losses == 0 are excluded (no observed transitions at all).
    Cached after first load.
    """
    if path not in _ACR_HYBRID_CACHE:
        _, ecoli_total_bl, _ = _load_kde()
        df = pd.read_csv(path)
        df = df[(df['gain_subsize'] > 0) & (df['loss_subsize'] > 0)].copy()
        df['lam'] = (df['gains'] + df['losses']) / ecoli_total_bl
        df = df[df['lam'] > 0].copy()
        g_m = df['gains_flow'] / df['gain_subsize_marginal']
        l_m = df['losses_flow'] / df['loss_subsize_marginal']
        gl  = g_m + l_m
        df['pi'] = np.where(gl > 0, g_m / gl, 0.5)
        _ACR_HYBRID_CACHE[path] = df[['lam', 'pi']].reset_index(drop=True)
    return _ACR_HYBRID_CACHE[path]


def _load_acr_marginal(path='pastmlout_marginal.csv'):
    """Load ACR data for the marginal rate model.

    Both π and λ are derived from marginal (flow-based) rates:

        g_marg = gains_flow / gain_subsize_marginal
        l_marg = losses_flow / loss_subsize_marginal
        π      = g_marg / (g_marg + l_marg)
        λ      = g_marg + l_marg

    Genes where either marginal rate is zero are excluded.
    Cached after first load.
    """
    if path not in _ACR_MARGINAL_CACHE:
        df = pd.read_csv(path)
        df = df[(df['gain_subsize'] > 0) & (df['loss_subsize'] > 0)].copy()
        g_m = df['gains_flow'] / df['gain_subsize_marginal']
        l_m = df['losses_flow'] / df['loss_subsize_marginal']
        df['g_m'] = g_m
        df['l_m'] = l_m
        df = df[(df['g_m'] > 0) & (df['l_m'] > 0)].copy()
        gl = df['g_m'] + df['l_m']
        df['lam'] = gl
        df['pi']  = df['g_m'] / gl
        _ACR_MARGINAL_CACHE[path] = df[['lam', 'pi']].reset_index(drop=True)
    return _ACR_MARGINAL_CACHE[path]


def _load_kde(path='scripts/kde_model.pkl'):
    """Load KDE model from disk, caching after first load.

    Returns (kde, ecoli_total_bl, n_ecoli_taxa).
    Supports both the new dict format (from build_kde.py) and the legacy
    raw-KDE format (ecoli_total_bl and n_ecoli_taxa will be None for legacy files).
    """
    if path not in _KDE_CACHE:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            _KDE_CACHE[path] = (obj['kde'], obj['ecoli_total_bl'], obj.get('n_ecoli_taxa'))
        else:
            # Legacy format: raw KDE object, no scaling metadata
            _KDE_CACHE[path] = (obj, None, None)
    return _KDE_CACHE[path]


def _compute_bl_stats(t):
    """
    Compute IQR-filtered mean branch length for a tree.
    Returns (upper_bound, bl_mean) where bl_mean is the mean over
    branches with length <= upper_bound (IQR-based filter in log10 space).
    """
    branch_lengths = np.array([n.dist for n in t.traverse() if not n.is_root() and n.dist > 0])
    log_bl = np.log10(branch_lengths)
    Q1, Q3 = np.percentile(log_bl, [25, 75])
    IQR = Q3 - Q1
    upper_bound = 10 ** (Q3 + 0.5 * IQR)
    bl_mean = np.mean(branch_lengths[branch_lengths <= upper_bound])
    return upper_bound, bl_mean

# ── π / λ / OR helpers ────────────────────────────────────────────────────────
 
def _stationary_from_or(pi1, pi2, OR):
    """Solve for the 4-state stationary distribution given marginal
    prevalences and an odds ratio.
 
    States use the same bit encoding as the rest of the file:
      0 = (trait1=0, trait2=0)
      1 = (trait1=1, trait2=0)   [state & 1]
      2 = (trait1=0, trait2=1)   [(state & 2) >> 1]
      3 = (trait1=1, trait2=1)
 
    OR = p11 * p00 / (p10 * p01)
 
    Derivation
    ----------
    Substituting p10 = pi1 - p11, p01 = pi2 - p11, p00 = 1 - pi1 - pi2 + p11
    into the OR definition and rearranging gives the Cornfield quadratic:
 
      (1 - OR) * p11^2
      + (1 - pi1 - pi2 + OR*(pi1 + pi2)) * p11
      - OR * pi1 * pi2
      = 0
 
    The physically valid root is the one in (max(0, pi1+pi2-1), min(pi1, pi2)).
    For OR > 1 we take the larger root (more co-occurrence than independence);
    for OR < 1 we take the smaller root.  OR == 1 reduces to independence:
    p11 = pi1 * pi2.
 
    Parameters
    ----------
    pi1, pi2 : float — marginal prevalences in (0, 1)
    OR       : float — target odds ratio; must be > 0
 
    Returns
    -------
    stat : np.ndarray shape (4,) — stationary probabilities [p00, p10, p01, p11]
    """
    lo = max(0.0, pi1 + pi2 - 1.0) + 1e-9
    hi = min(pi1, pi2) - 1e-9
 
    if hi <= lo:
        # Degenerate case: marginals leave no room — return independence
        p11 = float(np.clip(pi1 * pi2, 0.0, 1.0))
    elif abs(OR - 1.0) < 1e-10:
        p11 = pi1 * pi2
    else:
        a = 1.0 - OR
        b = 1.0 - pi1 - pi2 + OR * (pi1 + pi2)
        c = -OR * pi1 * pi2
        disc = max(b ** 2 - 4.0 * a * c, 0.0)
        x1 = (-b + np.sqrt(disc)) / (2.0 * a)
        x2 = (-b - np.sqrt(disc)) / (2.0 * a)
        # Both roots may or may not be in the valid range
        valid = sorted([x for x in (x1, x2) if lo <= x <= hi])
        if valid:
            # OR > 1 → positive association → want larger p11
            # OR < 1 → negative association → want smaller p11
            p11 = valid[-1] if OR > 1.0 else valid[0]
        else:
            # Fallback: independence, clamped
            p11 = float(np.clip(pi1 * pi2, lo, hi))
 
    p11 = float(np.clip(p11, lo, hi))
    p10 = pi1 - p11
    p01 = pi2 - p11
    p00 = 1.0 - pi1 - pi2 + p11
 
    stat = np.array([p00, p10, p01, p11], dtype=float)
    stat = np.clip(stat, 1e-12, None)
    stat /= stat.sum()
    return stat
 
 
def _build_Q_reversible(stat, lam):
    """Build a reversible 4-state Q matrix from a stationary distribution.
 
    Construction
    ------------
    Setting Q[i, j] = k * stat[j] for connected pairs (i, j) satisfies
    detailed balance exactly:
 
        stat[i] * Q[i, j] = k * stat[i] * stat[j] = stat[j] * Q[j, i]
 
    Only single-bit transitions are permitted (no simultaneous flips):
        0 ↔ 1 : trait1 changes, trait2 = 0 background  (00 ↔ 10)
        0 ↔ 2 : trait2 changes, trait1 = 0 background  (00 ↔ 01)
        1 ↔ 3 : trait2 changes, trait1 = 1 background  (10 ↔ 11)
        2 ↔ 3 : trait1 changes, trait2 = 1 background  (01 ↔ 11)
 
    The overall scale k is chosen so that the stationary-distribution-weighted
    mean exit rate equals lam (the scaled total rate λ from the π/λ
    decomposition), giving a physically interpretable rate magnitude.
 
    Parameters
    ----------
    stat : array-like shape (4,) — stationary distribution [p00, p10, p01, p11]
    lam  : float — target mean exit rate (= scaled λ for the target tree)
 
    Returns
    -------
    Q : np.ndarray shape (4, 4) — generator matrix (rows sum to 0)
    """
    stat = np.asarray(stat, dtype=float)
    Q = np.zeros((4, 4))
 
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        Q[i, j] = stat[j]
        Q[j, i] = stat[i]
 
    # Stationary-weighted mean exit rate before scaling
    raw_exit = float(np.dot(stat, Q.sum(axis=1)))
    if raw_exit > 0.0:
        Q *= lam / raw_exit
 
    # Set diagonal so each row sums to zero
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q
 
 
# ── Main simulation function ──────────────────────────────────────────────────

def synth_mutual_4state_nosim_p(dir, t, mod,
                                kde=None, bl_stats=None,
                                n_ecoli_taxa=None,
                                gamma_alpha=None):
    """Simulate a pair of binary traits under a 4-state CTMC model.

    Non-marginal (π from gains/gain_subsize) rate path — preserved for reference.
    See synth_mutual_4state_nosim for the current hybrid_m default.

    New default path: π / λ / OR parameterisation
    ---------------------------------------------
    Effect size is encoded as an odds ratio derived from mod:
 
        OR_base = 10^(2 * mod)
 
    and direction controls the sign:
 
        dir =  1  →  positive association  (OR = OR_base)
        dir =  0  →  no association        (OR = 1)
        dir = -1  →  negative association  (OR = 1 / OR_base)
 
    Rate sampling from the empirical ACR data is decomposed into two
    orthogonal quantities that are scaled independently:
 
        π   = g / (g + l)   — equilibrium prevalence; tree-invariant; never scaled
        λ   = g + l          — total rate;  scaled by ecoli_total_bl / target_total_bl
 
    The 4-state stationary distribution is solved analytically from
    (π1, π2, effective_OR) via the Cornfield quadratic so that marginal
    prevalences are guaranteed at stationarity regardless of mod or tree
    topology.  A reversible Q matrix is then built from this stationary
    distribution and scaled to λ.
 
    This decouples three sources of variation:
        • prevalence  ← π  (empirical distribution, tree-invariant)
        • effect size ← OR (set by mod, independent of rates)
        • transitions ← λ  (scaled to tree, controls D-statistic)
 
    Legacy KDE path
    ---------------
    If kde is not None the function falls back to the original
    total_bl_ntaxa scaling method for backward compatibility.
 
    Parameters
    ----------
    dir          : int, -1 / 0 / 1 — direction of association
    t            : ete3.Tree
    mod          : float — effect-size modifier; OR_base = 10^(2*mod)
    kde          : KDE object or None; if supplied uses legacy path
    bl_stats     : (upper_bound, bl_mean) from _compute_bl_stats (optional)
    n_ecoli_taxa : int — kept for signature compatibility; unused in new path
    gamma_alpha  : float or None — if set, branch lengths are gamma-scaled
 
    Returns
    -------
    lineages  : np.ndarray shape (n_leaves, 2) — trait1, trait2 per leaf
    prev      : np.ndarray shape (2,)           — tip prevalences
    gain_rates: list[float, float]              — scaled gain rates used
    loss_rates: list[float, float]              — scaled loss rates used
    zeros     : np.ndarray shape (2,)           — placeholder (legacy compat)
    leaves    : list[ete3.Node]                 — leaf nodes in sim order
    """
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []
 
    # Always load tree metadata from KDE pkl (cached; fast after first call)
    _, ecoli_total_bl, _loaded_n_ecoli = _load_kde()
 
    if bl_stats is None:
        bl_stats = _compute_bl_stats(t)
    upper_bound, bl_mean = bl_stats
 
    # IQR-filtered total branch length of the target tree
    total_bl = sum(
        min(n.dist, upper_bound)
        for n in t.traverse()
        if not n.is_root() and n.dist > 0
    )
 
    # ── Branch: legacy KDE path ───────────────────────────────────────────────
    if kde is not None:
        samples = kde.resample(2)
        gains_raw = 10 ** samples[0]
        losses_raw = 10 ** samples[1]
        root_state_bits = np.rint(samples[2]).astype(int)
        root_state = int(f'{root_state_bits[1]:0b}{root_state_bits[0]:0b}', 2)
 
        n_target_taxa = len(t.get_leaves())
        if n_ecoli_taxa is None:
            n_ecoli_taxa = _loaded_n_ecoli or n_target_taxa
        if ecoli_total_bl is not None and ecoli_total_bl > 0 and total_bl > 0:
            gain_rates = gains_raw * (ecoli_total_bl / total_bl) * (n_target_taxa / n_ecoli_taxa)
            loss_rates = losses_raw * (ecoli_total_bl / total_bl) * (n_target_taxa / n_ecoli_taxa)
        else:
            gain_rates = gains_raw * 0.00603 / bl_mean
            loss_rates = losses_raw * 0.00603 / bl_mean
 
        gain_modifier = [10 ** (-mod), 1, 10 ** (mod)]
        loss_modifier = [10 ** (mod),  1, 10 ** (-mod)]
 
        Q = np.zeros((4, 4))
        unnorm_rates = {
            0: {1: gain_rates[0], 2: gain_rates[1]},
            1: {0: loss_rates[0], 3: gain_rates[1] * gain_modifier[dir + 1]},
            2: {0: loss_rates[1], 3: gain_rates[0] * gain_modifier[dir + 1]},
            3: {1: loss_rates[1] * loss_modifier[dir + 1],
                2: loss_rates[0] * loss_modifier[dir + 1]},
        }
        for i in range(4):
            for j in unnorm_rates[i]:
                Q[i, j] = unnorm_rates[i][j]
            Q[i, i] = -np.sum(Q[i, :])
 
    # ── Branch: π / λ / OR path (default) ────────────────────────────────────
    else:
        df_acr = _load_acr()
 
        # Sample two independent (g, l) pairs from the empirical ACR distribution
        rows = df_acr.sample(n=2, replace=True)
        g_raw = rows['g'].values.astype(float)
        l_raw = rows['l'].values.astype(float)
 
        # Decompose into π (prevalence) and λ (total rate)
        lam_raw = g_raw + l_raw                # shape (2,)
        pi      = g_raw / lam_raw              # shape (2,); tree-invariant
 
        # Scale λ to the target tree; π is never modified
        f          = (ecoli_total_bl / total_bl
                      if (ecoli_total_bl and total_bl > 0) else 1.0)
        lam_scaled = lam_raw * f               # shape (2,)
 
        # Reconstruct gain/loss rates for output and legacy callers
        gain_rates = pi          * lam_scaled  # shape (2,)
        loss_rates = (1.0 - pi) * lam_scaled   # shape (2,)
 
        # Convert mod → effective OR; direction controls sign
        #   dir= 1 → OR = 10^(2*mod)       (positive association)
        #   dir= 0 → OR = 1                 (independence)
        #   dir=-1 → OR = 10^(-2*mod)       (negative association)
        OR_base      = 10.0 ** (2.0 * mod)
        effective_OR = float(OR_base ** dir)   # 1.0 when dir=0
 
        # Clip π values to avoid numerically degenerate stationary distributions
        pi1 = float(np.clip(pi[0], 0.02, 0.98))
        pi2 = float(np.clip(pi[1], 0.02, 0.98))
 
        # Solve for the 4-state stationary distribution analytically
        stat = _stationary_from_or(pi1, pi2, effective_OR)
 
        # Use geometric mean of the two traits' λ as the Q-matrix scale
        # (preserves the overall rate magnitude symmetrically across traits)
        lam_mean = float(np.sqrt(lam_scaled[0] * lam_scaled[1]))
 
        # Build reversible Q matrix satisfying detailed balance
        Q = _build_Q_reversible(stat, lam_mean)
 
        # Root state drawn from stationary distribution
        root_state = int(np.random.choice(4, p=stat))
 
    # ── CTMC simulation (shared by both paths) ────────────────────────────────
    sim = np.zeros(num_nodes, dtype=int)
    sim[node_map[t]] = root_state
 
    for node in t.traverse(strategy='preorder'):
        if node.up is None:
            continue
 
        curr_state = sim[node_map[node.up]]
        branch_t   = node.dist
 
        if gamma_alpha is not None and branch_t > 0:
            branch_t = branch_t * np.random.gamma(gamma_alpha, 1.0 / gamma_alpha)
 
        t_remaining = branch_t
 
        while t_remaining > 0:
            rate_out = -Q[curr_state, curr_state]
            if rate_out <= 0:
                break
            wait_time = np.random.exponential(1.0 / rate_out)
            if wait_time >= t_remaining:
                break
            t_remaining -= wait_time
            probs = Q[curr_state, :].copy()
            probs[curr_state] = 0.0
            prob_sum = probs.sum()
            if prob_sum <= 0:
                break
            probs /= prob_sum
            curr_state = int(np.random.choice(4, p=probs))
 
        sim[node_map[node]] = curr_state
        if node.is_leaf():
            leaves.append(node)
 
    # Decode 4-state simulation back to two binary traits
    trait1 = (sim & 1).astype(np.int16)
    trait2 = ((sim & 2) >> 1).astype(np.int16)
 
    lineages = np.stack(
        [trait1[[node_map[l] for l in leaves]],
         trait2[[node_map[l] for l in leaves]]],
        axis=1,
    )
    prev = lineages.mean(axis=0)
 
    return lineages, prev, list(gain_rates), list(loss_rates), np.zeros(2), leaves


def synth_mutual_4state_nosim(dir, t, mod,
                              kde=None, bl_stats=None,
                              n_ecoli_taxa=None,
                              gamma_alpha=None,
                              mix_p=0.5):
    """Simulate a pair of binary traits under a 4-state CTMC model.

    Default path: 50/50 mix of marginal and hybrid_m rate models
    -------------------------------------------------------------
    Each trait in a pair independently draws its (π, λ) from one of two pools:

        marginal  (prob = mix_p):
            π = g_marg / (g_marg + l_marg),  λ = g_marg + l_marg
            → higher λ → more transitions → higher D (random/independent)

        hybrid_m  (prob = 1 - mix_p):
            π = g_marg / (g_marg + l_marg),  λ = (gains + losses) / ecoli_total_bl
            → includes zero-count genes → lower λ → lower / negative D (clade-driven)

    Because each trait draws independently, trait pairs can be drawn from
    different models, producing a wide D range while preserving the
    low-prevalence π distribution (median π ≈ 0.11) from both pools.

    Effect size is encoded as an odds ratio derived from mod:

        OR_base = 10^(2 * mod)

    and direction controls the sign:

        dir =  1  →  positive association  (OR = OR_base)
        dir =  0  →  no association        (OR = 1)
        dir = -1  →  negative association  (OR = 1 / OR_base)

    The 4-state stationary distribution is solved analytically from
    (π1, π2, effective_OR) via the Cornfield quadratic.  A reversible Q matrix
    is built from this stationary distribution and scaled to the geometric
    mean of the two traits' λ values.

    Legacy KDE path
    ---------------
    If kde is not None the function falls back to the original
    total_bl_ntaxa scaling method for backward compatibility.

    Parameters
    ----------
    dir          : int, -1 / 0 / 1 — direction of association
    t            : ete3.Tree
    mod          : float — effect-size modifier; OR_base = 10^(2*mod)
    kde          : KDE object or None; if supplied uses legacy path
    bl_stats     : (upper_bound, bl_mean) from _compute_bl_stats (optional)
    n_ecoli_taxa : int — kept for signature compatibility; unused in new path
    gamma_alpha  : float or None — if set, branch lengths are gamma-scaled
    mix_p        : float in [0, 1] — probability each trait draws from the
                   marginal pool; default 0.5 (50/50 mix)

    Returns
    -------
    lineages  : np.ndarray shape (n_leaves, 2) — trait1, trait2 per leaf
    prev      : np.ndarray shape (2,)           — tip prevalences
    gain_rates: list[float, float]              — scaled gain rates used
    loss_rates: list[float, float]              — scaled loss rates used
    zeros     : np.ndarray shape (2,)           — placeholder (legacy compat)
    leaves    : list[ete3.Node]                 — leaf nodes in sim order
    """
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []

    # Always load tree metadata from KDE pkl (cached; fast after first call)
    _, ecoli_total_bl, _loaded_n_ecoli = _load_kde()

    if bl_stats is None:
        bl_stats = _compute_bl_stats(t)
    upper_bound, bl_mean = bl_stats

    # IQR-filtered total branch length of the target tree
    total_bl = sum(
        min(n.dist, upper_bound)
        for n in t.traverse()
        if not n.is_root() and n.dist > 0
    )

    # ── Branch: legacy KDE path ───────────────────────────────────────────────
    if kde is not None:
        samples = kde.resample(2)
        gains_raw = 10 ** samples[0]
        losses_raw = 10 ** samples[1]
        root_state_bits = np.rint(samples[2]).astype(int)
        root_state = int(f'{root_state_bits[1]:0b}{root_state_bits[0]:0b}', 2)

        n_target_taxa = len(t.get_leaves())
        if n_ecoli_taxa is None:
            n_ecoli_taxa = _loaded_n_ecoli or n_target_taxa
        if ecoli_total_bl is not None and ecoli_total_bl > 0 and total_bl > 0:
            gain_rates = gains_raw * (ecoli_total_bl / total_bl) * (n_target_taxa / n_ecoli_taxa)
            loss_rates = losses_raw * (ecoli_total_bl / total_bl) * (n_target_taxa / n_ecoli_taxa)
        else:
            gain_rates = gains_raw * 0.00603 / bl_mean
            loss_rates = losses_raw * 0.00603 / bl_mean

        gain_modifier = [10 ** (-mod), 1, 10 ** (mod)]
        loss_modifier = [10 ** (mod),  1, 10 ** (-mod)]

        Q = np.zeros((4, 4))
        unnorm_rates = {
            0: {1: gain_rates[0], 2: gain_rates[1]},
            1: {0: loss_rates[0], 3: gain_rates[1] * gain_modifier[dir + 1]},
            2: {0: loss_rates[1], 3: gain_rates[0] * gain_modifier[dir + 1]},
            3: {1: loss_rates[1] * loss_modifier[dir + 1],
                2: loss_rates[0] * loss_modifier[dir + 1]},
        }
        for i in range(4):
            for j in unnorm_rates[i]:
                Q[i, j] = unnorm_rates[i][j]
            Q[i, i] = -np.sum(Q[i, :])

    # ── Branch: mix of marginal and hybrid_m pools (default) ─────────────────
    else:
        f = (ecoli_total_bl / total_bl
             if (ecoli_total_bl and total_bl > 0) else 1.0)

        # Each trait independently draws from marginal (prob=mix_p) or hybrid_m
        lam_raw = np.empty(2)
        pi      = np.empty(2)
        for i in range(2):
            pool = (_load_acr_marginal() if np.random.rand() < mix_p
                    else _load_acr_hybrid_m())
            row       = pool.sample(n=1).iloc[0]
            lam_raw[i] = float(row['lam'])
            pi[i]      = float(row['pi'])

        # Scale λ to the target tree; π is never modified
        lam_scaled = lam_raw * f

        # Reconstruct gain/loss rates for output and legacy callers
        gain_rates = pi          * lam_scaled
        loss_rates = (1.0 - pi) * lam_scaled

        # Convert mod → effective OR; direction controls sign
        OR_base      = 10.0 ** (2.0 * mod)
        effective_OR = float(OR_base ** dir)   # 1.0 when dir=0

        # Clip π values to avoid numerically degenerate stationary distributions
        pi1 = float(np.clip(pi[0], 0.02, 0.98))
        pi2 = float(np.clip(pi[1], 0.02, 0.98))

        # Solve for the 4-state stationary distribution analytically
        stat = _stationary_from_or(pi1, pi2, effective_OR)

        # Use geometric mean of the two traits' λ as the Q-matrix scale
        lam_mean = float(np.sqrt(lam_scaled[0] * lam_scaled[1]))

        # Build reversible Q matrix satisfying detailed balance
        Q = _build_Q_reversible(stat, lam_mean)

        # Root state drawn from stationary distribution
        root_state = int(np.random.choice(4, p=stat))

    # ── CTMC simulation (shared by both paths) ────────────────────────────────
    sim = np.zeros(num_nodes, dtype=int)
    sim[node_map[t]] = root_state

    for node in t.traverse(strategy='preorder'):
        if node.up is None:
            continue

        curr_state = sim[node_map[node.up]]
        branch_t   = node.dist

        if gamma_alpha is not None and branch_t > 0:
            branch_t = branch_t * np.random.gamma(gamma_alpha, 1.0 / gamma_alpha)

        t_remaining = branch_t

        while t_remaining > 0:
            rate_out = -Q[curr_state, curr_state]
            if rate_out <= 0:
                break
            wait_time = np.random.exponential(1.0 / rate_out)
            if wait_time >= t_remaining:
                break
            t_remaining -= wait_time
            probs = Q[curr_state, :].copy()
            probs[curr_state] = 0.0
            prob_sum = probs.sum()
            if prob_sum <= 0:
                break
            probs /= prob_sum
            curr_state = int(np.random.choice(4, p=probs))

        sim[node_map[node]] = curr_state
        if node.is_leaf():
            leaves.append(node)

    # Decode 4-state simulation back to two binary traits
    trait1 = (sim & 1).astype(np.int16)
    trait2 = ((sim & 2) >> 1).astype(np.int16)

    lineages = np.stack(
        [trait1[[node_map[l] for l in leaves]],
         trait2[[node_map[l] for l in leaves]]],
        axis=1,
    )
    prev = lineages.mean(axis=0)

    return lineages, prev, list(gain_rates), list(loss_rates), np.zeros(2), leaves


def synth_mutual_4state_nosim_pilamold(dir, t, mod,
                              kde=None, bl_stats=None,
                              n_ecoli_taxa=None,
                              gamma_alpha=None):
    """
    Simulate a pair of binary traits under a 4-state CTMC model (no simultaneous transitions).

    Parameters
    ----------
    dir          : -1, 0, or 1 — direction of association
    t            : ete3 Tree
    mod          : effect size modifier (log10 scale)
    kde          : pre-loaded KDE object (optional; loaded from disk if None)
    bl_stats     : (upper_bound, bl_mean) tuple from _compute_bl_stats (optional)
    n_ecoli_taxa : int — number of leaves in the E. coli reference tree;
                   loaded from KDE pkl if None.
    gamma_alpha  : float — if set, branch lengths are gamma-scaled
    """
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []

    # Always load tree metadata from KDE pkl (cached; KDE object itself unused here)
    _, ecoli_total_bl, _loaded_n_ecoli = _load_kde()
    if n_ecoli_taxa is None:
        n_ecoli_taxa = _loaded_n_ecoli

    if bl_stats is None:
        bl_stats = _compute_bl_stats(t)
    upper_bound, bl_mean = bl_stats

    # IQR-filtered total branch length of the target tree
    total_bl = sum(
        min(n.dist, upper_bound)
        for n in t.traverse()
        if not n.is_root() and n.dist > 0
    )

    if kde is not None:
        # ── Legacy KDE path ────────────────────────────────────────────────────
        # Sample rates from the joint KDE trained on gains_flow / ecoli_total_bl.
        # Scales by total_bl_ntaxa correction (original behaviour).
        samples = kde.resample(2)
        gains_raw = 10 ** samples[0]
        losses_raw = 10 ** samples[1]
        root_state_bits = np.rint(samples[2]).astype(int)
        root_state = int(f'{root_state_bits[1]:0b}{root_state_bits[0]:0b}', 2)

        n_target_taxa = len(t.get_leaves())
        if n_ecoli_taxa is None:
            n_ecoli_taxa = n_target_taxa
        if ecoli_total_bl is not None and ecoli_total_bl > 0 and total_bl > 0:
            gain_rates = gains_raw * (ecoli_total_bl / total_bl) * (n_target_taxa / n_ecoli_taxa)
            loss_rates = losses_raw * (ecoli_total_bl / total_bl) * (n_target_taxa / n_ecoli_taxa)
        else:
            gain_rates = gains_raw * 0.00603 / bl_mean
            loss_rates = losses_raw * 0.00603 / bl_mean

    else:
        # ── Direct ACR path (default) ──────────────────────────────────────────
        # Rates: gains / gain_subsize  and  losses / loss_subsize.
        # Scaling: π/λ decomposition — π = g/(g+l) is preserved; λ = g+l is
        # scaled by ecoli_total_bl / target_total_bl.  This was the best
        # cross-tree scaling method in the rate-variant exploration.
        df_acr = _load_acr()
        rows = df_acr.sample(n=2, replace=True)
        gains_raw = rows['g'].values.astype(float)
        losses_raw = rows['l'].values.astype(float)

        lam_raw = gains_raw + losses_raw + 1e-30
        pi = gains_raw / lam_raw
        f = (ecoli_total_bl / total_bl) if (ecoli_total_bl is not None and total_bl > 0) else 1.0
        lam_scaled = lam_raw * f
        gain_rates = pi * lam_scaled
        loss_rates = (1.0 - pi) * lam_scaled

        # Root state from stationary distribution of the two traits' π values
        p0 = (1.0 - pi[0]) * (1.0 - pi[1])
        p1 = pi[0]          * (1.0 - pi[1])
        p2 = (1.0 - pi[0]) * pi[1]
        p3 = pi[0]          * pi[1]
        probs = np.array([p0, p1, p2, p3])
        probs /= probs.sum()
        root_state = int(np.random.choice(4, p=probs))

    gain_modifier = [10 ** (-mod), 1, 10 ** (mod)]
    loss_modifier = [10 ** (mod), 1, 10 ** (-mod)]

    # Build rate matrix (Q)
    Q = np.zeros((4, 4))
    unnorm_rates = {
        0: {1: gain_rates[0], 2: gain_rates[1]},
        1: {0: loss_rates[0], 3: gain_rates[1] * gain_modifier[dir + 1]},
        2: {0: loss_rates[1], 3: gain_rates[0] * gain_modifier[dir + 1]},
        3: {1: loss_rates[1] * loss_modifier[dir + 1], 2: loss_rates[0] * loss_modifier[dir + 1]},
    }

    for i in range(4):
        for j in unnorm_rates[i]:
            Q[i, j] = unnorm_rates[i][j]
        Q[i, i] = -np.sum(Q[i, :])  # row sums to 0

    sim = np.zeros((num_nodes, 1), dtype=int)
    sim[node_map[t]] = root_state

    for node in t.traverse(strategy='preorder'):
        if node.up is None:
            continue

        parent_state = sim[node_map[node.up], 0]
        curr_state = parent_state
        branch_t = node.dist
        if gamma_alpha is not None and branch_t > 0:
            branch_t = branch_t * np.random.gamma(gamma_alpha, 1.0 / gamma_alpha)
        t_remaining = branch_t

        while t_remaining > 0:
            rate_out = -Q[curr_state, curr_state]
            if rate_out <= 0:
                break

            wait_time = np.random.exponential(1 / rate_out)
            if wait_time >= t_remaining:
                break

            t_remaining -= wait_time
            probs = Q[curr_state, :].copy()
            probs[curr_state] = 0
            probs /= probs.sum()
            curr_state = np.random.choice(4, p=probs)

        sim[node_map[node], 0] = curr_state
        if node.is_leaf():
            leaves.append(node)

    trait1 = (sim[:, 0] & 1).astype(np.int16)
    trait2 = ((sim[:, 0] & 2) >> 1).astype(np.int16)

    lineages = np.stack([trait1[[node_map[l] for l in leaves]],
                         trait2[[node_map[l] for l in leaves]]], axis=1)
    prev = lineages.mean(axis=0)

    return lineages, prev, gain_rates.tolist(), loss_rates.tolist(), np.zeros(2), leaves

def synth_directional(dir, t, mod, kde=None, bl_stats=None):
    """
    Simulate a directional (asymmetric) pair of traits under a 4-state CTMC model.

    Parameters
    ----------
    dir      : -1, 0, or 1 — direction of association
    t        : ete3 Tree
    mod      : effect size modifier (log10 scale)
    kde      : pre-loaded KDE object (optional; loaded from disk if None)
    bl_stats : (upper_bound, bl_mean) tuple from _compute_bl_stats (optional)
    """
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []

    if kde is None:
        kde, _ecoli_total_bl, _n_ecoli = _load_kde()

    samples = kde.resample(2)
    gains = 10**samples[0]
    losses = 10**samples[1]
    root_state_bits = np.rint(samples[2]).astype(int)
    root_state = int(f'{root_state_bits[1]:0b}{root_state_bits[0]:0b}', 2)

    if bl_stats is None:
        bl_stats = _compute_bl_stats(t)
    upper_bound, _ = bl_stats

    total_bl = sum(
        min(n.dist, upper_bound)
        for n in t.traverse()
        if not n.is_root() and n.dist > 0
    )

    _, bl_mean = bl_stats
    gain_rates = gains * 0.00603 / bl_mean
    loss_rates = losses * 0.00603 / bl_mean

    gain_modifier = [10 ** (-mod), 1, 10 ** (mod)]
    loss_modifier = [10 ** (mod), 1, 10 ** (-mod)]

    # Build rate matrix (Q)
    Q = np.zeros((4, 4))
    unnorm_rates = {
        0: {1: gain_rates[0], 2: gain_rates[1]},
        1: {0: loss_rates[0], 3: gain_rates[1] * gain_modifier[dir + 1]},
        2: {0: loss_rates[1], 3: gain_rates[0]},
        3: {1: loss_rates[1] * loss_modifier[dir + 1], 2: loss_rates[0]},
    }

    for i in range(4):
        for j in unnorm_rates[i]:
            Q[i, j] = unnorm_rates[i][j]
        Q[i, i] = -np.sum(Q[i, :])  # row sums to 0

    sim = np.zeros((num_nodes, 1), dtype=int)
    sim[node_map[t]] = root_state

    for node in t.traverse(strategy='preorder'):
        if node.up is None:
            continue

        parent_state = sim[node_map[node.up], 0]
        curr_state = parent_state
        t_remaining = node.dist

        while t_remaining > 0:
            rate_out = -Q[curr_state, curr_state]
            if rate_out <= 0:
                break

            wait_time = np.random.exponential(1 / rate_out)
            if wait_time >= t_remaining:
                break

            t_remaining -= wait_time
            probs = Q[curr_state, :].copy()
            probs[curr_state] = 0
            probs /= probs.sum()
            curr_state = np.random.choice(4, p=probs)

        sim[node_map[node], 0] = curr_state
        if node.is_leaf():
            leaves.append(node)

    trait1 = (sim[:, 0] & 1).astype(np.int16)
    trait2 = ((sim[:, 0] & 2) >> 1).astype(np.int16)

    lineages = np.stack([trait1[[node_map[l] for l in leaves]],
                         trait2[[node_map[l] for l in leaves]]], axis=1)
    prev = lineages.mean(axis=0)

    return lineages, prev, gains.tolist(), losses.tolist(), np.zeros(2), leaves


# 4-STATE MODEL IMPLEMENTATION WITH SIMULTANEOUS TRANSITIONS
def synth_mutual_4state(dir, t, mod):
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []

    #From ecoli pangenome (5-95%) learned log normal distribution of rates

    gain_mean = 0.512
    gain_std = 0.470

    loss_mean = 1.012
    loss_std = 0.490

    gains = np.random.lognormal(mean=gain_mean, sigma=gain_std, size = 2)
    losses = np.random.lognormal(mean=loss_mean, sigma=loss_std, size = 2)


    # Normalize gain/loss rates by total branch length
    bl = sum(sorted([i.dist for i in t.traverse()])[:-3])
    gain_rates = gains * 6.019839999999989 / bl# changing branch length units form ecoli tree to current tree
    loss_rates = losses * 6.019839999999989 / bl#/ (bl * MULTIPLIER)

    # Define gain/loss modifiers
    gain_modifier = [10 ** (-mod), 1, 10 ** (mod)]
    loss_modifier = [10 ** (mod), 1, 10 ** (-mod)]

    # Unnormalized rates for 4-state model transitions (states 0-3)
    unnorm_rates = {
        0: {
            1: gain_rates[0],
            2: gain_rates[1],
            3: (gain_rates[0] * gain_rates[1]) / (gain_rates[0] + gain_rates[1] + 1e-12) * gain_modifier[dir + 1]  # Avoid division by zero
        },
        1: {
            0: loss_rates[0],
            3: gain_rates[1] * gain_modifier[dir + 1],
            2: (loss_rates[0] * gain_rates[1]) / (loss_rates[0] + gain_rates[1] + 1e-12)
        },
        2: {
            0: loss_rates[1],
            3: gain_rates[0] * gain_modifier[dir + 1],
            1: (loss_rates[1] * gain_rates[0]) / (loss_rates[1] + gain_rates[0] + 1e-12)
        },
        3: {
            1: loss_rates[1] * loss_modifier[dir + 1],
            2: loss_rates[0] * loss_modifier[dir + 1],
            0: (loss_rates[0] * loss_rates[1]) / (loss_rates[0] + loss_rates[1] + 1e-12) * loss_modifier[dir + 1]
        }
    }


    # Normalize so total rates per state equal sum of single transitions only
    rates = {}
    for state, transitions in unnorm_rates.items():
        single_trans_sum = sum([rate for k, rate in transitions.items() if bin(k).count("1") - bin(state).count("1") == 1 or bin(k).count("1") - bin(state).count("1") == -1])
        total_sum = sum(transitions.values())
        if total_sum == 0:
            rates[state] = transitions
            continue
        normalization_factor = single_trans_sum / total_sum if total_sum != 0 else 1.0
        rates[state] = {k: v * normalization_factor for k, v in transitions.items()}

    sim = np.zeros((num_nodes, 1), dtype=int)
    sim[node_map[t]] = np.random.choice([0,1])

    for node in t.traverse(strategy='preorder'):
        if node.up is None:
            continue

        parent_state = sim[node_map[node.up], 0]
        curr_rates = rates[parent_state]

        dist = node.dist# * MULTIPLIER
        total_rate = sum(curr_rates.values())
        prob_change = 1 - np.exp(-total_rate * dist)

        if np.random.rand() < prob_change and total_rate > 0:
            next_states = list(curr_rates.keys())
            probs = np.array([curr_rates[s] for s in next_states]) / total_rate
            new_state = np.random.choice(next_states, p=probs)
        else:
            new_state = parent_state

        sim[node_map[node], 0] = new_state
        if node.is_leaf():
            leaves.append(node)

    # Decode states to binary traits
    trait1 = (sim[:, 0] & 1).astype(np.int16)
    trait2 = ((sim[:, 0] & 2) >> 1).astype(np.int16)

    lineages = np.stack([trait1[[node_map[l] for l in leaves]],
                         trait2[[node_map[l] for l in leaves]]], axis=1)
    prev = lineages.mean(axis=0)

    gains_out = gains.tolist()
    losses_out = losses.tolist()
    dists = np.zeros(2)

    return lineages, prev, gains_out, losses_out, dists, leaves

# 4-STATE MODEL IMPLEMENTATION WITHOUT SIMULTANEOUS TRANSITIONS
def synth_mutual_4state_nosim2(dir, t, mod):
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []

    #From ecoli pangenome (5-95%) learned log normal distribution of rates

    # gain_mean = 0.512
    # gain_std = 0.470

    # loss_mean = 1.012
    # loss_std = 0.490

    # gains = np.random.lognormal(mean=gain_mean, sigma=gain_std, size = 2)
    # losses = np.random.lognormal(mean=loss_mean, sigma=loss_std, size = 2)

    with open('scripts/kde_model.pkl', 'rb') as f:
        kde = pickle.load(f)

    samples = kde.resample(2)
    gains = 10**samples[0]
    losses = 10**samples[1]
    root_state = np.rint(samples[2]).astype(int)


    # Normalize gain/loss rates by total branch length
    # bl = sum(sorted([i.dist for i in t.traverse()])[:-3])
    # gain_rates = gains * 6.019839999999989 / bl# changing branch length units form ecoli tree to current tree
    # loss_rates = losses * 6.019839999999989 / bl#/ (bl * MULTIPLIER)

    # median branch length scaling:
    bl = np.median(np.array([i.dist for i in t.traverse()]))
    gain_rates = gains * 0.00103 / bl
    loss_rates = losses * 0.00103 / bl

    # Define gain/loss modifiers
    gain_modifier = [10 ** (-mod), 1, 10 ** (mod)]
    loss_modifier = [10 ** (mod), 1, 10 ** (-mod)]

    # Rates for 4-state model transitions (states 0-3)
    rates = {
        0: {
            1: gain_rates[0],
            2: gain_rates[1],
        },
        1: {
            0: loss_rates[0],
            3: gain_rates[1] * gain_modifier[dir + 1],
        },
        2: {
            0: loss_rates[1],
            3: gain_rates[0] * gain_modifier[dir + 1],
        },
        3: {
            1: loss_rates[1] * loss_modifier[dir + 1],
            2: loss_rates[0] * loss_modifier[dir + 1],
        }
    }

    sim = np.zeros((num_nodes, 1), dtype=int)
    sim[node_map[t]] = root_state

    for node in t.traverse(strategy='preorder'):
        if node.up is None:
            continue

        parent_state = sim[node_map[node.up], 0]
        curr_rates = rates[parent_state]

        dist = node.dist
        total_rate = sum(curr_rates.values())
        prob_change = 1 - np.exp(-total_rate * dist)

        if np.random.rand() < prob_change and total_rate > 0:
            next_states = list(curr_rates.keys())
            probs = np.array([curr_rates[s] for s in next_states]) / total_rate
            new_state = np.random.choice(next_states, p=probs)
        else:
            new_state = parent_state

        sim[node_map[node], 0] = new_state
        if node.is_leaf():
            leaves.append(node)

    # Decode states to binary traits
    trait1 = (sim[:, 0] & 1).astype(np.int16)
    trait2 = ((sim[:, 0] & 2) >> 1).astype(np.int16)

    lineages = np.stack([trait1[[node_map[l] for l in leaves]],
                         trait2[[node_map[l] for l in leaves]]], axis=1)
    prev = lineages.mean(axis=0)

    gains_out = gains.tolist()
    losses_out = losses.tolist()
    dists = np.zeros(2)

    return lineages, prev, gains_out, losses_out, dists, leaves

# MULTIVARIATE BROWNIAN MOTION FOR SIMULATION OF TRAITS USING COVARIANCE OF INTERACTIONS STRENGTH
def synth_mutual_mvBM(dir, t, interaction_strength, prev1  = 50, prev2 = 50):
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []
    # with open('scripts/kde_mvBM_model.pkl', 'rb') as f:
    #     kde = pickle.load(f)
    # while True:
    #     samples = kde.resample(2)[0]
    #     if np.all(samples < 0.95) and np.all(samples > 0.05):
    #         break
    # prev1 = samples[0]*100
    # prev2 = samples[1]*100

    # Simulate two traits under multivariate Brownian motion
    trait_continuous = np.zeros((num_nodes, 2))  # two traits at each node

    trait_continuous[node_map[t], :] = np.zeros(2) #np.random.normal(0, 1, size=2)
    bl = sum(node.dist for node in t.traverse())
    var_trait1 = 1.0/bl
    var_trait2 = 1.0/bl
    epsilon = 1e-6  # small buffer to prevent singular matrix
    max_covar = np.sqrt(var_trait1 * var_trait2) * (1 - epsilon)

    # Allow dir = -1, 0, 1 and scale by interaction_strength
    raw_covar = interaction_strength * dir * np.sqrt(var_trait1 * var_trait2)

    # Clip covar to be within safe bounds
    covar = np.clip(raw_covar, -max_covar, max_covar)
    # covar = interaction_strength * dir * np.sqrt(var_trait1 * var_trait2)
    cov_matrix = np.array([
        [var_trait1, covar],
        [covar, var_trait2]
    ])

    # Cholesky decomposition for efficient sampling
    chol_cov = np.linalg.cholesky(cov_matrix)

    # Traverse tree and evolve traits
    for node in t.traverse(strategy='preorder'):
        if node.up is None:
            continue

        parent_idx = node_map[node.up]
        current_idx = node_map[node]

        # Brownian motion increment proportional to branch length
        variance_scale = node.dist
        increment = np.random.normal(0, 1, size=2)
        increment = chol_cov @ increment * np.sqrt(variance_scale)

        trait_continuous[current_idx] = trait_continuous[parent_idx] + increment

    # Discretize continuous traits if desired (quartile thresholds)
    upper_q1 = np.percentile(trait_continuous[:, 0], 100 - prev1)
    upper_q2 = np.percentile(trait_continuous[:, 1], 100 - prev2)

    # Example binary traits: 1 if above median, else 0
    trait1 = (trait_continuous[:, 0] >= upper_q1).astype(np.int16)
    trait2 = (trait_continuous[:, 1] >= upper_q2).astype(np.int16)

    # Collect the simulated traits
    sim = np.zeros((num_nodes, 2), dtype=np.int16)
    sim[:, 0] = trait1
    sim[:, 1] = trait2


    # Collect leaves
    for node in t.traverse():
        if node.is_leaf():
            leaves.append(node)

    leaf_indices = [node_map[l] for l in leaves]
    lineages = sim[leaf_indices, :]

    prev = lineages.mean(axis=0)

    # Placeholder for compatibility
    gains = [np.nan, np.nan]
    losses = [np.nan, np.nan]
    dists = np.zeros(2)

    return lineages, prev, gains, losses, dists, leaves

def synth_asym(dir, t, mod=1.0, kde_path="scripts/kde_model.pkl"):

    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []

    # === Sample parameters from KDE ===
    with open(kde_path, "rb") as f:
        kde = pickle.load(f)
    samples = kde.resample(1)
    gains = 10**samples[0]
    losses = 10**samples[1]
    root_state = np.rint(samples[2]).astype(int)

    # branch length scaling
    branch_lengths = np.array([n.dist for n in t.traverse() if not n.is_root() and n.dist > 0])
    log_bl = np.log10(branch_lengths)
    Q1, Q3 = np.percentile(log_bl, [25, 75])
    IQR = Q3 - Q1
    log_upper_bound = Q3 + 0.5 * IQR
    upper_bound = 10 ** log_upper_bound
    bl = np.mean(branch_lengths[branch_lengths <= upper_bound])
    gain_rate = gains[0] * 0.00603 / bl
    loss_rate = losses[0] * 0.00603 / bl

    def simulate_trait(tree, gain_rate, loss_rate, root_state):
        tree = tree.copy()
        tree.add_features(state=root_state)
        for node in tree.traverse("preorder"):
            if node.is_root():
                continue
            parent_state = node.up.state
            p_gain = 1 - np.exp(-gain_rate * node.dist)
            p_loss = 1 - np.exp(-loss_rate * node.dist)
            if parent_state == 0:
                node.add_features(state=np.random.binomial(1, p_gain))
            else:
                node.add_features(state=1 - np.random.binomial(1, p_loss))
        return tree

    tX = simulate_trait(t, gain_rate, loss_rate, root_state)
    leaves = [leaf for leaf in tX.iter_leaves()]
    X = np.array([leaf.state for leaf in tX.iter_leaves()])

    beta = dir * abs(mod)
    intercept = np.log(np.mean(X) / (1 - np.mean(X) + 1e-9))  
    pY = expit(intercept + beta * X)
    Y = np.random.binomial(1, pY)

    lineages = np.stack([X, Y], axis=1)
    prev = lineages.mean(axis=0)
    zeros = np.zeros(2)

    return lineages, prev, gains.tolist(), losses.tolist(), zeros, leaves


def synth_mutual(dir, t, mod):
    MULTIPLIER = 1e12
    NUM_TRIALS = 1
    num_nodes = len(t.get_descendants())+1
    TREE_DISTS = {}
    # gains = np.random.randint(10, 20, size=2) #from quartiles of ecoli pdes >5% prev
    # losses = np.random.randint(10, 20, size=2)
    # gains = np.random.randint(6, 22, size=2) #from quartiles of ecoli pdes >5% prev
    # losses = np.random.randint(14, 131, size=2)
    gains = (np.random.exponential(1/75.36528854308317, size = 2) * num_nodes).astype(int) #rate params from approximate exponetioal of ecoli pangeneome form 5%-95% 
    losses = (np.random.exponential(1/26.029638058875307,size = 2) * num_nodes).astype(int)
    # gains = np.array([5,5])
    # losses = np.array([50,50])
    dists = np.zeros(2)
    bl = sum([i.dist for i in t.traverse()]) # type: ignore
    gain_rates = gains/(bl*MULTIPLIER)
    loss_rates = losses/(bl*MULTIPLIER)


    gain_modifier = [10**(-mod),1,10**(mod)]
    loss_modifier = [10**(mod),1,10**(-mod)]

    num_nodes = len(t.get_descendants())+1
    num_traits = len(gain_rates)
    sim = np.zeros((num_nodes,num_traits,NUM_TRIALS), dtype = np.int16)
    node_map = {node:ind for ind,node in enumerate([t] + t.get_descendants())}
    leaves = []

    for node in t.traverse(): # type: ignore
        if node.up == None:
            continue
        parent = sim[node_map[node.up],:,:]
        gain_array = np.array([])
        loss_array = np.array([])
        dist = node.get_distance(t)

        for g,l, idx in zip(gain_rates,loss_rates, range(num_traits)):
            if parent[0, 0] and parent[1, 0]:
                l *= loss_modifier[dir+1]
            if parent[1, 0] or parent[0,0]:
                g *= gain_modifier[dir+1]

            gain_events = np.random.binomial(node.dist*MULTIPLIER, g,NUM_TRIALS)
            loss_events = np.random.binomial(node.dist*MULTIPLIER, l,NUM_TRIALS)

            if gain_array.shape[0] == 0:
                gain_array = gain_events
                gain_array = np.expand_dims(gain_array, axis=0)
                loss_array = loss_events
                loss_array = np.expand_dims(loss_array, axis=0)
            else:
                gain_array = np.concatenate((gain_array,np.expand_dims(gain_events, axis=0)), axis = 0)
                loss_array = np.concatenate((loss_array,np.expand_dims(loss_events, axis=0)), axis = 0)
        # gain_mask = np.logical_not(np.logical_xor(parent,gain_array))
        # gain_array[gain_mask] = 0
        # loss_mask = np.logical_xor(parent,loss_array)
        # loss_array[loss_mask] = 0
        sim[node_map[node],:,:] = parent + gain_array - loss_array
        sim[node_map[node],:,:][sim[node_map[node],:,:] > 1] = 1
        sim[node_map[node],:,:][sim[node_map[node],:,:] < 0] = 0
        if node.is_leaf(): leaves.append(node)


    lineages = sim[[node_map[node] for node in leaves],:,:]
    prev = (lineages[:,:,0] > 0).mean(axis = 0)
    return lineages, prev, gains, losses, dists, leaves
