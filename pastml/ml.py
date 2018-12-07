import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pastml.models.f81_like import is_f81_like, get_f81_pij, get_mu, F81, EFT
from pastml.models.hky import get_hky_pij, KAPPA, HKY
from pastml.models.jtt import get_jtt_pij, JTT_FREQUENCIES, JTT
from pastml.parsimony import parsimonious_acr, MP
from pastml import get_personalized_feature_name, CHARACTER, STATES, METHOD, NUM_SCENARIOS, NUM_UNRESOLVED_NODES, \
    NUM_NODES, NUM_TIPS, NUM_STATES_PER_NODE, PERC_UNRESOLVED

CHANGES_PER_AVG_BRANCH = 'state_changes_per_avg_branch'
SCALING_FACTOR = 'scaling_factor'
FREQUENCIES = 'frequencies'
LOG_LIKELIHOOD = 'log_likelihood'
RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR = '{}_restricted_{{}}'.format(LOG_LIKELIHOOD)

JOINT = 'JOINT'
MPPA = 'MPPA'
MAP = 'MAP'
ALL = 'ALL'
ML = 'ML'

MARGINAL_PROBABILITIES = 'marginal_probabilities'

MODEL = 'model'

MIN_VALUE = np.log10(np.finfo(np.float64).eps)
MAX_VALUE = np.log10(np.finfo(np.float64).max)

MARGINAL_ML_METHODS = {MPPA, MAP}
ML_METHODS = MARGINAL_ML_METHODS | {JOINT}
META_ML_METHODS = {ML, ALL}


BU_LH = 'BOTTOM_UP_LIKELIHOOD'
TD_LH = 'TOP_DOWN_LIKELIHOOD'
LH = 'LIKELIHOOD'
LH_SF = 'LIKELIHOOD_SF'
BU_LH_SF = 'BOTTOM_UP_LIKELIHOOD_SF'
BU_LH_JOINT_STATES = 'BOTTOM_UP_LIKELIHOOD_JOINT_STATES'
TD_LH_SF = 'TOP_DOWM_LIKELIHOOD_SF'
ALLOWED_STATES = 'ALLOWED_STATES'
JOINT_STATE = 'JOINT_STATE'


def is_marginal(method):
    """
    Checks if the method is marginal, i.e. MAP, MPPA, or one of the meta-methods (ALL, ML).

    :param method: prediction method
    :type method: str
    :return: bool
    """
    return method in MARGINAL_ML_METHODS or method in META_ML_METHODS


def is_ml(method):
    """
    Checks if the method is max likelihood, i.e. JOINT or one of the marginal ones.

    :param method: prediction method
    :type method: str
    :return: bool
    """
    return method in ML_METHODS or method in META_ML_METHODS


def is_meta_ml(method):
    """
    Checks if the method is a meta max likelihood method, combining several methods, i.e. ML or ALL.

    :param method: prediction method
    :type method: str
    :return: bool
    """
    return method in META_ML_METHODS


def get_default_ml_method():
    return MPPA


def get_pij_method(model=F81, frequencies=None, kappa=None):
    """
    Returns a function for calculation of probability matrix of substitutions i->j over time t.

    :param kappa: kappa parameter for HKY model
    :type kappa: float
    :param frequencies: array of state frequencies \pi_i
    :type frequencies: numpy.array
    :param model: model of character evolution
    :type model: str
    :return: probability matrix
    :rtype: function
    """
    if is_f81_like(model):
        mu = get_mu(frequencies)
        return lambda t: get_f81_pij(t, frequencies, mu)
    if JTT == model:
        return get_jtt_pij
    if HKY == model:
        return lambda t: get_hky_pij(t, frequencies, kappa)


def get_bottom_up_likelihood(tree, character, frequencies, sf, kappa=None, is_marginal=True, model=F81):
    """
    Calculates the bottom-up likelihood for the given tree.
    The likelihood for each node is stored in the corresponding feature,
    given by get_personalised_feature_name(feature, BU_LH).

    :param model: model of character evolution
    :type model: str
    :param is_marginal: whether the likelihood reconstruction is marginal (true) or joint (false)
    :type is_marginal: bool
    :param tree: tree of interest
    :type tree: ete3.Tree
    :param character: character for which the likelihood is calculated
    :type character: str
    :param frequencies: array of state frequencies \pi_i
    :type frequencies: numpy.array
    :param sf: scaling factor
    :type sf: float
    :return: log likelihood
    :rtype: float
    """
    lh_sf_feature = get_personalized_feature_name(character, BU_LH_SF)
    lh_feature = get_personalized_feature_name(character, BU_LH)
    lh_joint_state_feature = get_personalized_feature_name(character, BU_LH_JOINT_STATES)
    allowed_state_feature = get_personalized_feature_name(character, ALLOWED_STATES)

    get_pij = get_pij_method(model, frequencies, kappa)
    for node in tree.traverse('postorder'):
        likelihood_array = np.ones(len(frequencies), dtype=np.float64) * getattr(node, allowed_state_feature)
        factors = 0
        for child in node.children:
            child_likelihoods = get_pij(child.dist * sf) * getattr(child, lh_feature)
            if is_marginal:
                child_likelihoods = child_likelihoods.sum(axis=1)
            else:
                child_states = child_likelihoods.argmax(axis=1)
                child.add_feature(lh_joint_state_feature, child_states)
                child_likelihoods = child_likelihoods.max(axis=1)

            factors += rescale(child_likelihoods, fraction_of_limit=len(node.children))
            likelihood_array *= child_likelihoods

        if np.all(likelihood_array == 0):
            return -np.inf

        factors += rescale(likelihood_array, fraction_of_limit=len(node.up.children) if not node.is_root() else 1)
        node.add_feature(lh_feature, likelihood_array)
        node.add_feature(lh_sf_feature, factors + sum(getattr(_, lh_sf_feature) for _ in node.children))
    root_likelihoods = getattr(tree, lh_feature) * frequencies
    root_likelihoods = root_likelihoods.sum() if is_marginal else root_likelihoods.max()
    return np.log(root_likelihoods) - getattr(tree, lh_sf_feature) * np.log(10)


def rescale(likelihood_array, fraction_of_limit):
    """
    Rescales the likelihood array if it gets too small/large, by multiplying it by a factor of 10.
    :param fraction_of_limit: int, to be rescaled the min (max) non-zero likelihood value should be
    smaller that MIN_VALUE / fraction_of_limit (larger than MAX_LIMIT / fraction_of_limit).
    :param likelihood_array: numpy array containing the likelihood to be rescaled
    :return: float, factor of 10 by which the likelihood array has been multiplies.
    """

    max_limit = MAX_VALUE / fraction_of_limit
    min_limit = MIN_VALUE / fraction_of_limit

    min_lh_value = np.log10(np.min(likelihood_array[np.nonzero(likelihood_array)]))
    max_lh_value = np.log10(np.max(likelihood_array[np.nonzero(likelihood_array)]))

    factors = 0
    if max_lh_value > max_limit:
        factors = max_limit - max_lh_value - 1
        likelihood_array *= np.power(10, factors)
    elif min_lh_value < min_limit:
        factors = min(-min_lh_value, max_limit - max_lh_value)
        likelihood_array *= np.power(10, factors)
    return factors


def optimize_likelihood_params(tree, character, frequencies, sf, kappa, avg_br_len,
                               optimise_sf=True, optimise_frequencies=True, optimise_kappa=True,
                               model=F81):
    """
    Optimizes the likelihood parameters (state frequencies and scaling factor) for the given tree.

    :param model: model of character evolution
    :type model: str
    :param avg_br_len: avg branch length
    :type avg_br_len: float
    :param tree: tree of interest
    :type tree: ete3.Tree
    :param character: character for which the likelihood is optimised
    :type character: str
    :param frequencies: array of initial state frequencies
    :type frequencies: numpy.array
    :param sf: initial scaling factor
    :type sf: float
    :param optimise_sf: whether the scaling factor needs to be optimised
    :type optimise_sf: bool
    :param optimise_frequencies: whether the state frequencies need to be optimised
    :type optimise_frequencies: bool
    :return: optimized parameters and log likelihood: ((frequencies, scaling_factor), optimum)
    :rtype: tuple
    """
    bounds = []
    if optimise_frequencies:
        bounds += [np.array([1e-6, 10e6], np.float64)] * (len(frequencies) - 1)
    if optimise_sf:
        bounds += [np.array([0.001 / avg_br_len, 10. / avg_br_len])]
    if optimise_kappa:
        bounds += [np.array([1e-6, 20.])]
    bounds = np.array(bounds, np.float64)

    def get_real_params_from_optimised(ps):
        freqs = frequencies
        if optimise_frequencies:
            freqs = np.hstack((ps[: (len(frequencies) - 1)], [1.]))
            freqs /= freqs.sum()
        sf_val = ps[(len(frequencies) - 1) if optimise_frequencies else 0] if optimise_sf else sf
        kappa_val = ps[((len(frequencies) - 1) if optimise_frequencies else 0) + (1 if optimise_sf else 0)] \
            if optimise_kappa else kappa
        return freqs, sf_val, kappa_val

    def get_v(ps):
        if np.any(pd.isnull(ps)):
            return np.nan
        freqs, sf_val, kappa_val = get_real_params_from_optimised(ps)
        res = get_bottom_up_likelihood(tree=tree, character=character, frequencies=freqs,
                                       sf=sf_val, kappa=kappa_val, is_marginal=True, model=model)
        return np.inf if pd.isnull(res) else -res

    for i in range(10):
        if i == 0:
            vs = np.hstack((frequencies[:-1] / frequencies[-1] if optimise_frequencies else [],
                            [sf] if optimise_sf else [],
                            [kappa] if optimise_kappa else []))
        else:
            vs = np.random.uniform(bounds[:, 0], bounds[:, 1])
        fres = minimize(get_v, x0=vs, method='L-BFGS-B', bounds=bounds)
        if fres.success and not np.any(np.isnan(fres.x)):
            return get_real_params_from_optimised(fres.x), -fres.fun


def calculate_top_down_likelihood(tree, character, frequencies, sf, kappa=None, model=F81):
    """
    Calculates the top-down likelihood for the given tree.
    The likelihood for each node is stored in the corresponding feature,
    given by get_personalised_feature_name(feature, TD_LH).

    To calculate the top-down likelihood of a node, we assume that the tree is rooted in this node
    and combine the likelihoods of the “up-subtrees”,
    e.g. to calculate the top-down likelihood of a node N1 being in a state i,
    given that its parent node is P and its brother node is N2, we imagine that the tree is re-rooted in N1,
    therefore P becoming the child of N1, and N2 its grandchild.
    We then calculate the bottom-up likelihood from the P subtree:
    L_top_down(N1, i) = \sum_j P(i -> j, dist(N1, P)) * L_top_down(P) * \sum_k P(j -> k, dist(N2, P)) * L_bottom_up (N2).

    For the root node we assume its top-down likelihood to be 1 for all the states.

    :param model: model of character evolution
    :type model: str
    :param sf: scaling factor
    :type sf: float
    :param character: character whose ancestral state likelihood is being calculated
    :type character: str
    :param tree: tree of interest (with bottom-up likelihood pre-calculated)
    :type tree: ete3.Tree
    :param frequencies: state frequencies
    :type frequencies: numpy.array
    :return: void, stores the node top-down likelihoods in the get_personalised_feature_name(feature, TD_LH) feature.
    """

    lh_feature = get_personalized_feature_name(character, TD_LH)
    lh_sf_feature = get_personalized_feature_name(character, TD_LH_SF)
    bu_lh_feature = get_personalized_feature_name(character, BU_LH)
    bu_lh_sf_feature = get_personalized_feature_name(character, BU_LH_SF)

    get_pij = get_pij_method(model, frequencies, kappa)
    for node in tree.traverse('preorder'):
        if node.is_root():
            node.add_feature(lh_feature, np.ones(len(frequencies), np.float64))
            node.add_feature(lh_sf_feature, 0)
            continue

        parent = node.up
        parent_bu_likelihood = getattr(parent, bu_lh_feature)

        node_pjis = np.transpose(get_pij(node.dist * sf))
        node_contribution = getattr(node, bu_lh_feature).dot(node_pjis)

        parent_likelihood = getattr(parent, lh_feature) * parent_bu_likelihood
        parent_likelihood[np.nonzero(parent_likelihood)] /= node_contribution[np.nonzero(parent_likelihood)]
        factors = getattr(parent, lh_sf_feature) + getattr(parent, bu_lh_sf_feature) - getattr(node, bu_lh_sf_feature)

        td_likelihood = parent_likelihood.dot(node_pjis)
        factors += rescale(td_likelihood, fraction_of_limit=len(node.children) if not node.is_leaf() else 1)

        node.add_feature(lh_feature, td_likelihood)
        node.add_feature(lh_sf_feature, factors)


def initialize_allowed_states(tree, feature, states):
    """
    Initializes the allowed state arrays for tips based on their states given by the feature.

    :param tree: tree for which the tip likelihoods are to be initialized
    :type tree: ete3.Tree
    :param feature: feature in which the tip states are stored
        (the value could be None for a missing state or list if multiple stated are possible)
    :type feature: str
    :param states: ordered array of states.
    :type states: numpy.array
    :return: void, adds the get_personalised_feature_name(feature, ALLOWED_STATES) feature to tree tips.
    """
    allowed_states_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    state2index = dict(zip(states, range(len(states))))

    for node in tree.traverse():
        node_states = getattr(node, feature, set())
        if not node_states:
            allowed_states = np.ones(len(state2index), dtype=np.int)
        else:
            allowed_states = np.zeros(len(state2index), dtype=np.int)
            for state in node_states:
                allowed_states[state2index[state]] = 1
        node.add_feature(allowed_states_feature, allowed_states)


def alter_zero_tip_allowed_states(tree, feature):
    """
    Alters the bottom-up likelihood arrays for zero-distance tips
    to make sure they do not contradict with other zero-distance tip siblings.

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the likelihood is altered
    :return: void, modifies the get_personalised_feature_name(feature, BU_LH) feature to zero-distance tips.
    """
    zero_parent2tips = defaultdict(list)

    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    for tip in tree:
        if tip.dist == 0:
            state = getattr(tip, feature, None)
            if state is not None and state != '':
                zero_parent2tips[tip.up].append(tip)

    # adjust zero tips to contain all the zero tip options as states
    for parent, zero_tips in zero_parent2tips.items():
        # If there is a common state do nothing
        counts = None
        for tip in zero_tips:
            if counts is None:
                counts = getattr(tip, allowed_state_feature).copy()
            else:
                counts += getattr(tip, allowed_state_feature)
        if counts.max() == len(zero_tips):
            continue

        # Otherwise set all tip states to state union
        allowed_states = None
        for tip in zero_tips:
            if allowed_states is None:
                allowed_states = getattr(tip, allowed_state_feature).copy()
            else:
                tip_allowed_states = getattr(tip, allowed_state_feature)
                allowed_states[np.nonzero(tip_allowed_states)] = 1
            tip.add_feature(allowed_state_feature, allowed_states)


def unalter_zero_tip_allowed_states(tree, feature, state2index):
    """
    Unalters the bottom-up likelihood arrays for zero-distance tips
    to contain ones only in their states.

    :param state2index: dict, mapping between states and their indices in the likelihood array
    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the likelihood was altered
    :return: void, modifies the get_personalised_feature_name(feature, BU_LH) feature to zero-distance tips.
    """
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    for tip in tree:
        if tip.dist > 0:
            continue
        state = getattr(tip, feature, set())
        if state:
            initial_allowed_states = np.zeros(len(state2index), np.int)
            for _ in state:
                initial_allowed_states[state2index[_]] = 1
            allowed_states = getattr(tip, allowed_state_feature) & initial_allowed_states
            tip.add_feature(allowed_state_feature, (allowed_states
                                                    if np.any(allowed_states > 0) else initial_allowed_states))


def unalter_zero_tip_joint_states(tree, feature, state2index):
    """
    Unalters the joint tip states for zero-distance tips
    to contain only their states.

    :param state2index: dict, mapping between states and their indices in the joint state array
    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the likelihood was altered
    :return: void, modifies the get_personalised_feature_name(feature, BU_LH_JOINT_STATES) feature to zero-distance tips.
    """
    lh_joint_state_feature = get_personalized_feature_name(feature, BU_LH_JOINT_STATES)
    for tip in tree:
        if tip.dist > 0:
            continue
        state = getattr(tip, feature, set())
        if len(state) > 1:
            allowed_indices = {state2index[_] for _ in state}
            allowed_index = next(iter(allowed_indices))
            joint_states = getattr(tip, lh_joint_state_feature)
            for i in range(len(state2index)):
                if joint_states[i] not in allowed_indices:
                    joint_states[i] = allowed_index
        elif len(state) == 1:
            tip.add_feature(lh_joint_state_feature, np.ones(len(state2index), np.int) * state2index[next(iter(state))])


def calculate_marginal_likelihoods(tree, feature, frequencies):
    """
    Calculates marginal likelihoods for each tree node
    by multiplying state frequencies with their bottom-up and top-down likelihoods.

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the likelihood is calculated
    :param frequencies: numpy array of state frequencies
    :return: void, stores the node marginal likelihoods in the get_personalised_feature_name(feature, LH) feature.
    """
    bu_lh_feature = get_personalized_feature_name(feature, BU_LH)
    bu_lh_sf_feature = get_personalized_feature_name(feature, BU_LH_SF)
    td_lh_feature = get_personalized_feature_name(feature, TD_LH)
    td_lh_sf_feature = get_personalized_feature_name(feature, TD_LH_SF)
    lh_feature = get_personalized_feature_name(feature, LH)
    lh_sf_feature = get_personalized_feature_name(feature, LH_SF)
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    for node in tree.traverse('preorder'):
        likelihood = getattr(node, bu_lh_feature) * getattr(node, td_lh_feature) * frequencies \
                     * getattr(node, allowed_state_feature)
        node.add_feature(lh_feature, likelihood)
        node.add_feature(lh_sf_feature, getattr(node, td_lh_sf_feature) + getattr(node, bu_lh_sf_feature))

        node.del_feature(bu_lh_feature)
        node.del_feature(bu_lh_sf_feature)
        node.del_feature(td_lh_feature)
        node.del_feature(td_lh_sf_feature)


def check_marginal_likelihoods(tree, feature):
    """
    Sanity check: combined bottom-up and top-down likelihood of each node of the tree must be the same.

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the likelihood is calculated
    :return: void, stores the node marginal likelihoods in the get_personalised_feature_name(feature, LH) feature.
    """
    lh_feature = get_personalized_feature_name(feature, LH)
    lh_sf_feature = get_personalized_feature_name(feature, LH_SF)

    for node in tree.traverse():
        if not node.is_root() and not (node.is_leaf() and node.dist == 0):
            node_loglh = np.log10(getattr(node, lh_feature).sum()) - getattr(node, lh_sf_feature)
            parent_loglh = np.log10(getattr(node.up, lh_feature).sum()) - getattr(node.up, lh_sf_feature)
            assert (round(node_loglh, 2) == round(parent_loglh, 2))


def convert_likelihoods_to_probabilities(tree, feature, states):
    """
    Normalizes each node marginal likelihoods to convert them to marginal probabilities.

    :param states: numpy array of states in the order corresponding to the marginal likelihood arrays
    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the probabilities are calculated
    :return: pandas DataFrame, that maps node names to their marginal likelihoods.
    """
    lh_feature = get_personalized_feature_name(feature, LH)

    name2probs = {}

    for node in tree.traverse():
        lh = getattr(node, lh_feature)
        name2probs[node.name] = lh / lh.sum()

    return pd.DataFrame.from_dict(name2probs, orient='index', columns=states)


def choose_ancestral_states_mppa(tree, feature, states, force_joint=True):
    """
    Chooses node ancestral states based on their marginal probabilities using MPPA method.

    :param force_joint: make sure that Joint state is chosen even if it has a low probability.
    :type force_joint: bool
    :param tree: tree of interest
    :type tree: ete3.Tree
    :param feature: character for which the ancestral states are to be chosen
    :type feature: str
    :param states: possible character states in order corresponding to the probabilities array
    :type states: numpy.array
    :return: number of ancestral scenarios selected,
        calculated by multiplying the number of selected states for all nodes.
        Also modified the get_personalized_feature_name(feature, ALLOWED_STATES) feature of each node
        to only contain the selected states.
    :rtype: int
    """
    lh_feature = get_personalized_feature_name(feature, LH)
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    joint_state_feature = get_personalized_feature_name(feature, JOINT_STATE)

    n = len(states)
    _, state2array = get_state2allowed_states(states, False)

    num_scenarios = 1
    unresolved_nodes = 0
    num_states = 0

    # If force_joint == True,
    # we make sure that the joint state is always chosen,
    # for this we sort the marginal probabilities array as [lowest_non_joint_mp, ..., highest_non_joint_mp, joint_mp]
    # select k in 1:n such as the correction between choosing 0, 0, ..., 1/k, ..., 1/k and our sorted array is min
    # and return the corresponding states
    for node in tree.traverse():
        marginal_likelihoods = getattr(node, lh_feature)
        marginal_probs = marginal_likelihoods / marginal_likelihoods.sum()
        if force_joint:
            joint_index = getattr(node, joint_state_feature)
            joint_prob = marginal_probs[joint_index]
            marginal_probs = np.hstack((np.sort(np.delete(marginal_probs, joint_index)), [joint_prob]))
        else:
            marginal_probs = np.sort(marginal_probs)
        best_k = n
        best_correstion = np.inf
        for k in range(1, n + 1):
            correction = np.hstack((np.zeros(n - k), np.ones(k) / k)) - marginal_probs
            correction = correction.dot(correction)
            if correction < best_correstion:
                best_correstion = correction
                best_k = k

        num_scenarios *= best_k
        num_states += best_k
        if force_joint:
            indices_selected = sorted(range(n),
                                      key=lambda _: (0 if n == joint_index else 1, -marginal_likelihoods[_]))[:best_k]
        else:
            indices_selected = sorted(range(n), key=lambda _: -marginal_likelihoods[_])[:best_k]
        if best_k == 1:
            allowed_states = state2array[indices_selected[0]]
        else:
            allowed_states = np.zeros(len(states), dtype=np.int)
            allowed_states[indices_selected] = 1
            unresolved_nodes += 1
        node.add_feature(allowed_state_feature, allowed_states)

    return num_scenarios, unresolved_nodes, num_states


def choose_ancestral_states_map(tree, feature, states):
    """
    Chooses node ancestral states based on their marginal probabilities using MAP method.

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the ancestral states are to be chosen
    :param states: numpy.array of possible character states in order corresponding to the probabilities array
    :return: void, modified the get_personalized_feature_name(feature, ALLOWED_STATES) feature of each node
        to only contain the selected states.
    """
    lh_feature = get_personalized_feature_name(feature, LH)
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    _, state2array = get_state2allowed_states(states, False)

    for node in tree.traverse():
        marginal_likelihoods = getattr(node, lh_feature)
        node.add_feature(allowed_state_feature, state2array[marginal_likelihoods.argmax()])


def choose_ancestral_states_joint(tree, feature, states, frequencies):
    """
    Chooses node ancestral states based on their marginal probabilities using joint method.

    :param frequencies: numpy array of state frequencies
    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the ancestral states are to be chosen
    :param states: numpy.array of possible character states in order corresponding to the probabilities array
    :return: void, modified the get_personalized_feature_name(feature, ALLOWED_STATES) feature of each node
        to only contain the selected states.
    """
    lh_feature = get_personalized_feature_name(feature, BU_LH)
    lh_state_feature = get_personalized_feature_name(feature, BU_LH_JOINT_STATES)
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    joint_state_feature = get_personalized_feature_name(feature, JOINT_STATE)
    _, state2array = get_state2allowed_states(states, False)

    def chose_consistent_state(node, state_index):
        node.add_feature(joint_state_feature, state_index)
        node.add_feature(allowed_state_feature, state2array[state_index])

        for child in node.children:
            chose_consistent_state(child, getattr(child, lh_state_feature)[state_index])

    chose_consistent_state(tree, (getattr(tree, lh_feature) * frequencies).argmax())


def get_state2allowed_states(states, by_name=True):
    # tips allowed state arrays won't be modified so we might as well just share them
    n = len(states)
    all_ones = np.ones(n, np.int)
    state2array = {}
    for index, state in enumerate(states):
        allowed_state_array = np.zeros(n, np.int)
        allowed_state_array[index] = 1
        state2array[state if by_name else index] = allowed_state_array
    if by_name:
        state2array[None] = all_ones
        state2array[''] = all_ones
    return all_ones, state2array


def ml_acr(tree, character, prediction_method, model, states, avg_br_len, num_nodes, num_tips, freqs=None, sf=None,
           kappa=None, force_joint=True):
    """
    Calculates ML states on the tree and stores them in the corresponding feature.

    :param states: numpy array of possible states
    :param prediction_method: str, MPPA (marginal approximation), MAP (max a posteriori) or JOINT
    :param tree: ete3.Tree, the tree of interest
    :param character: str, character for which the ML states are reconstructed
    :param model: str, evolutionary model, F81 (Felsenstein 81-like), JC (Jukes-Cantor-like) or EFT (estimate from tips)
    :param avg_br_len: float, average non-zero branch length of the tree.
    :param freqs: numpy array of predefined frequencies (or None if they are to be estimated)
    :param sf: float, predefined scaling factor (or None if it is to be estimated)
    :return: dict, mapping between reconstruction parameters and values
    """
    n = len(states)
    state2index = dict(zip(states, range(n)))
    missing_data = 0.
    observed_frequencies = np.zeros(n, np.float64)
    for _ in tree:
        state = getattr(_, character, set())
        if state:
            num_node_states = len(state)
            for _ in state:
                observed_frequencies[state2index[_]] += 1. / num_node_states
        else:
            missing_data += 1
    total_count = observed_frequencies.sum() + missing_data
    observed_frequencies /= observed_frequencies.sum()
    missing_data /= total_count

    logger = logging.getLogger('pastml')
    logger.debug('Observed frequencies for {}:{}{}.'
                 .format(character,
                         ''.join('\n\tfrequency of {}:\t{:.3f}'.format(state, observed_frequencies[
                             state2index[state]])
                                 for state in states),
                         '\n\tfraction of missing data:\t{:.3f}'.format(
                             missing_data) if missing_data else ''))

    if freqs is not None and model not in {F81, HKY}:
        logging.warning('Some frequencies were specified in the parameter file, '
                        'but the selected model ({}) ignores them. '
                        'Use F81 (or HKY for nucleotide characters only) '
                        'for taking user-specified frequencies into account.'.format(model))
    optimise_frequencies = model in {F81, HKY} and freqs is None
    if JTT == model:
        frequencies = JTT_FREQUENCIES
    elif EFT == model:
        frequencies = observed_frequencies
    elif model in {F81, HKY} and freqs is not None:
        frequencies = freqs
    else:
        frequencies = np.ones(n, dtype=np.float64) / n

    initialize_allowed_states(tree, character, states)
    alter_zero_tip_allowed_states(tree, character)
    if sf:
        optimise_sf = False
    else:
        sf = 1. / avg_br_len
        optimise_sf = True
    if HKY == model:
        if kappa:
            optimise_kappa = False
        else:
            optimise_kappa = True
            kappa = 4.
    else:
        optimise_kappa = False

    likelihood = get_bottom_up_likelihood(tree=tree, character=character, frequencies=frequencies, sf=sf, kappa=kappa,
                                          is_marginal=True, model=model)
    if not optimise_sf and not optimise_frequencies and not optimise_kappa:
        logger.debug('All the parameters are fixed for {}:{}{}{}{}.'
                     .format(character,
                             ''.join('\n\tfrequency of {}:\t{:.3f}'.format(state, frequencies[
                                 state2index[state]])
                                     for state in states),
                             '\n\tSF:\t{:.3f}, i.e. {:.3f} changes per avg branch'
                             .format(sf, sf * avg_br_len),
                             '\n\tkappa:\t{:.3f}'.format(kappa) if HKY == model else '',
                             '\n\tlog likelihood:\t{:.3f}'.format(likelihood)))
    else:
        logger.debug('Initial values for {} parameter optimisation:{}{}{}{}.'
                     .format(character,
                             ''.join('\n\tfrequency of {}:\t{:.3f}'.format(state, frequencies[
                                 state2index[state]])
                                     for state in states),
                             '\n\tSF:\t{:.3f}, i.e. {:.3f} changes per avg branch'
                             .format(sf, sf * avg_br_len),
                             '\n\tkappa:\t{:.3f}'.format(kappa) if HKY == model else '',
                             '\n\tlog likelihood:\t{:.3f}'.format(likelihood)))
        if optimise_sf:
            (_, sf, _), likelihood = optimize_likelihood_params(tree=tree, character=character, frequencies=frequencies,
                                                                sf=sf, kappa=kappa,
                                                                optimise_frequencies=False, optimise_sf=optimise_sf,
                                                                optimise_kappa=False, avg_br_len=avg_br_len,
                                                                model=model)
            if optimise_frequencies or optimise_kappa:
                logger.debug('Pre-optimised SF for {}:{}{}.'
                             .format(character,
                                     '\n\tSF:\t{:.3f}, i.e. {:.3f} changes per avg branch'
                                     .format(sf, sf * avg_br_len),
                                     '\n\tlog likelihood:\t{:.3f}'.format(likelihood)))
        if optimise_frequencies or optimise_kappa:
            (frequencies, sf, kappa), likelihood = \
                optimize_likelihood_params(tree=tree, character=character, frequencies=frequencies, sf=sf, kappa=kappa,
                                           optimise_frequencies=optimise_frequencies, optimise_sf=optimise_sf,
                                           optimise_kappa=optimise_kappa, avg_br_len=avg_br_len, model=model)

        logger.debug('Optimised {} values:{}{}{}{}'
                     .format(character,
                             ''.join('\n\tfrequency of {}:\t{:.3f}'.format(state, frequencies[
                                 state2index[state]])
                                     for state in states) if optimise_frequencies else '',
                             '\n\tSF:\t{:.3f}, i.e. {:.3f} changes per avg branch'
                             .format(sf, sf * avg_br_len),
                             '\n\tkappa:\t{:.3f}'.format(kappa) if HKY == model else '',
                             '\n\tlog likelihood:\t{:.3f}'.format(likelihood)))

    result = {LOG_LIKELIHOOD: likelihood, CHARACTER: character, METHOD: prediction_method, MODEL: model,
              FREQUENCIES: frequencies, SCALING_FACTOR: sf, CHANGES_PER_AVG_BRANCH: sf * avg_br_len, STATES: states,
              NUM_NODES: num_nodes, NUM_TIPS: num_tips}
    if HKY == model:
        result[KAPPA] = kappa

    results = []

    def process_reconstructed_states(method):
        if method == prediction_method or is_meta_ml(prediction_method):
            method_character = get_personalized_feature_name(character, method) \
                if prediction_method != method else character
            convert_allowed_states2feature(tree, character, states, method_character)
            res = result.copy()
            res[CHARACTER] = method_character
            res[METHOD] = method
            results.append(res)

    def process_restricted_likelihood_and_states(method):
        alter_zero_tip_allowed_states(tree, character)
        restricted_likelihood = get_bottom_up_likelihood(tree=tree, character=character,
                                                         frequencies=frequencies, sf=sf, kappa=kappa,
                                                         is_marginal=True, model=model)
        unalter_zero_tip_allowed_states(tree, character, state2index)
        note_restricted_likelihood(method, restricted_likelihood)
        process_reconstructed_states(method)

    def note_restricted_likelihood(method, restricted_likelihood):
        logger.debug('Log likelihood for {} after {} state selection:\t{:.3f}'
                     .format(character, method, restricted_likelihood))
        result[RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(method)] = restricted_likelihood

    if prediction_method != MAP:
        # Calculate joint restricted likelihood
        restricted_likelihood = get_bottom_up_likelihood(tree=tree, character=character,
                                                         frequencies=frequencies, sf=sf, kappa=kappa,
                                                         is_marginal=False, model=model)
        note_restricted_likelihood(JOINT, restricted_likelihood)
        unalter_zero_tip_joint_states(tree, character, state2index)
        choose_ancestral_states_joint(tree, character, states, frequencies)
        process_reconstructed_states(JOINT)

    if is_marginal(prediction_method):
        initialize_allowed_states(tree, character, states)
        alter_zero_tip_allowed_states(tree, character)
        get_bottom_up_likelihood(tree=tree, character=character, frequencies=frequencies, sf=sf, kappa=kappa,
                                 is_marginal=True, model=model)
        calculate_top_down_likelihood(tree, character, frequencies, sf, kappa=kappa, model=model)
        unalter_zero_tip_allowed_states(tree, character, state2index)
        calculate_marginal_likelihoods(tree, character, frequencies)
        # check_marginal_likelihoods(tree, feature)
        result[MARGINAL_PROBABILITIES] = convert_likelihoods_to_probabilities(tree, character, states)

        choose_ancestral_states_map(tree, character, states)
        process_restricted_likelihood_and_states(MAP)

        if MPPA == prediction_method or is_meta_ml(prediction_method):

            if ALL == prediction_method:
                pars_acr_results = parsimonious_acr(tree, character, MP, states, num_nodes, num_tips)
                results.extend(pars_acr_results)
                for pars_acr_res in pars_acr_results:
                    _parsimonious_states2allowed_states(tree, pars_acr_res[CHARACTER], character, state2index)
                    alter_zero_tip_allowed_states(tree, character)
                    restricted_likelihood = get_bottom_up_likelihood(tree=tree, character=character,
                                                                     frequencies=frequencies, sf=sf, kappa=kappa,
                                                                     is_marginal=True, model=model)
                    note_restricted_likelihood(pars_acr_res[METHOD], restricted_likelihood)

            result[NUM_SCENARIOS], result[NUM_UNRESOLVED_NODES], result[NUM_STATES_PER_NODE] = \
                choose_ancestral_states_mppa(tree, character, states, force_joint=force_joint)
            result[NUM_STATES_PER_NODE] /= num_nodes
            result[PERC_UNRESOLVED] = result[NUM_UNRESOLVED_NODES] * 100 / num_nodes
            logger.debug('{} node{} unresolved ({:.2f}%) for {} by {}, '
                         'i.e. {:.4f} state{} per node in average.'
                         .format(result[NUM_UNRESOLVED_NODES], 's are' if result[NUM_UNRESOLVED_NODES] != 1 else ' is',
                                 result[PERC_UNRESOLVED], character, MPPA,
                                 result[NUM_STATES_PER_NODE], 's' if result[NUM_STATES_PER_NODE] > 1 else ''))
            process_restricted_likelihood_and_states(MPPA)

    return results


def convert_allowed_states2feature(tree, feature, states, out_feature=None):
    if out_feature is None:
        out_feature = feature
    allowed_states_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    for node in tree.traverse():
        node.add_feature(out_feature, set(states[getattr(node, allowed_states_feature).astype(bool)]))


def _parsimonious_states2allowed_states(tree, ps_feature, feature, state2index):
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    for node in tree.traverse():
        pars_states = getattr(node, ps_feature)
        allowed_states = np.zeros(len(state2index), dtype=int)
        for state in pars_states:
            allowed_states[state2index[state]] = 1
        node.add_feature(allowed_state_feature, allowed_states)
