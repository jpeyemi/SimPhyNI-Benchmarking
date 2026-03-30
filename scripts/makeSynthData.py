import numpy as np
from ete3 import Tree
import pandas as pd
import pickle
from scipy.stats import gaussian_kde
from scipy.special import expit

# Module-level KDE cache: avoids reloading from disk on every call
_KDE_CACHE = {}


def _load_kde(path='scripts/kde_model.pkl'):
    """Load KDE model from disk, caching after first load."""
    if path not in _KDE_CACHE:
        with open(path, 'rb') as f:
            _KDE_CACHE[path] = pickle.load(f)
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


def synth_mutual_4state_nosim(dir, t, mod, kde=None, bl_stats=None):
    """
    Simulate a pair of binary traits under a 4-state CTMC model (no simultaneous transitions).

    Parameters
    ----------
    dir      : -1, 0, or 1 — direction of association
    t        : ete3 Tree
    mod      : effect size modifier (log10 scale)
    kde      : pre-loaded KDE model (optional; loaded from disk if None)
    bl_stats : (upper_bound, bl_mean) tuple from _compute_bl_stats (optional; computed if None)
    """
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []

    if kde is None:
        kde = _load_kde()

    samples = kde.resample(2)
    gains = 10**samples[0]
    losses = 10**samples[1]
    root_state_bits = np.rint(samples[2]).astype(int)
    root_state = int(f'{root_state_bits[1]:0b}{root_state_bits[0]:0b}', 2)

    if bl_stats is None:
        _, bl = _compute_bl_stats(t)
    else:
        _, bl = bl_stats

    gain_rates = gains * 0.00603 / bl
    loss_rates = losses * 0.00603 / bl

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

def synth_directional(dir, t, mod, kde=None, bl_stats=None):
    """
    Simulate a directional (asymmetric) pair of traits under a 4-state CTMC model.

    Parameters
    ----------
    dir      : -1, 0, or 1 — direction of association
    t        : ete3 Tree
    mod      : effect size modifier (log10 scale)
    kde      : pre-loaded KDE model (optional; loaded from disk if None)
    bl_stats : (upper_bound, bl_mean) tuple from _compute_bl_stats (optional; computed if None)
    """
    num_nodes = len(t.get_descendants()) + 1
    node_map = {node: idx for idx, node in enumerate([t] + t.get_descendants())}
    leaves = []

    if kde is None:
        kde = _load_kde()

    samples = kde.resample(2)
    gains = 10**samples[0]
    losses = 10**samples[1]
    root_state_bits = np.rint(samples[2]).astype(int)
    root_state = int(f'{root_state_bits[1]:0b}{root_state_bits[0]:0b}', 2)

    if bl_stats is None:
        _, bl = _compute_bl_stats(t)
    else:
        _, bl = bl_stats

    gain_rates = gains * 0.00603 / bl
    loss_rates = losses * 0.00603 / bl

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
