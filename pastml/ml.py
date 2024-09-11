import logging
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pastml import get_personalized_feature_name, CHARACTER, METHOD, NUM_SCENARIOS, NUM_UNRESOLVED_NODES, \
    NUM_STATES_PER_NODE, PERC_UNRESOLVED, STATES
from pastml.models import ModelWithFrequencies
from pastml.parsimony import parsimonious_acr, MP

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
STATE_COUNTS = 'STATE_COUNTS'
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


def get_bottom_up_loglikelihood(tree, character, model, is_marginal=True, alter=True):
    """
    Calculates the bottom-up loglikelihood for the given tree.
    The likelihood for each node is stored in the corresponding feature,
    given by get_personalised_feature_name(feature, BU_LH).

    :param model: model of character evolution
    :type model: pastml.models.Model
    :param is_marginal: whether the likelihood reconstruction is marginal (true) or joint (false)
    :type is_marginal: bool
    :param tree: tree of interest
    :type tree: ete3.Tree
    :param character: character for which the likelihood is calculated
    :type character: str
    :return: log likelihood
    :rtype: float
    """

    altered_nodes = []
    if 0 == model.tau and alter:
        altered_nodes = alter_zero_node_allowed_states(tree, character)

    lh_sf_feature = get_personalized_feature_name(character, BU_LH_SF)
    lh_feature = get_personalized_feature_name(character, BU_LH)
    lh_joint_state_feature = get_personalized_feature_name(character, BU_LH_JOINT_STATES)
    allowed_state_feature = get_personalized_feature_name(character, ALLOWED_STATES)

    get_pij = model.get_Pij_t
    for node in tree.traverse('postorder'):
        calc_node_bu_likelihood(node, allowed_state_feature, lh_feature, lh_sf_feature, lh_joint_state_feature,
                                is_marginal, get_pij)
    root_likelihoods = getattr(tree, lh_feature) * model.frequencies
    root_likelihoods = root_likelihoods.sum() if is_marginal else root_likelihoods.max()

    if altered_nodes:
        if is_marginal:
            unalter_zero_node_allowed_states(altered_nodes, character)
        else:
            unalter_zero_node_joint_states(altered_nodes, character)

    return np.log(root_likelihoods) - getattr(tree, lh_sf_feature) / np.log10(np.e)


def calc_node_bu_likelihood(node, allowed_state_feature, lh_feature, lh_sf_feature, lh_joint_state_feature, is_marginal,
                            get_pij):
    allowed_states = getattr(node, allowed_state_feature)
    log_likelihood_array = np.log10(np.ones(len(allowed_states), dtype=np.float64) * allowed_states)
    factors = 0
    for child in node.children:
        child_likelihoods = get_pij(child.dist) * getattr(child, lh_feature)
        if is_marginal:
            child_likelihoods = child_likelihoods.sum(axis=1)
        else:
            child_states = child_likelihoods.argmax(axis=1)
            child.add_feature(lh_joint_state_feature, child_states)
            child_likelihoods = child_likelihoods.max(axis=1)
        child_likelihoods = np.maximum(child_likelihoods, 0)
        log_likelihood_array += np.log10(child_likelihoods)
        if np.all(log_likelihood_array == -np.inf):
            raise PastMLLikelihoodError("The parent node {} and its child node {} have non-intersecting states, "
                                        "and are connected by a zero-length ({:g}) branch. "
                                        "This creates a zero likelihood value. "
                                        "To avoid this issue check the restrictions on these node states "
                                        "and/or use a smoothing factor (tau)."
                                        .format(node.name, child.name, child.dist))
        factors += rescale_log(log_likelihood_array)
    node.add_feature(lh_feature, np.power(10, log_likelihood_array))
    node.add_feature(lh_sf_feature, factors + sum(getattr(_, lh_sf_feature) for _ in node.children))


def rescale_log(loglikelihood_array):
    """
    Rescales the likelihood array if it gets too small/large, by multiplying it by a factor of 10.
    :param loglikelihood_array: numpy array containing the loglikelihood to be rescaled
    :return: float, factor of 10 by which the likelihood array has been multiplies.
    """

    max_limit = MAX_VALUE
    min_limit = MIN_VALUE

    non_zero_loglh_array = loglikelihood_array[loglikelihood_array > -np.inf]
    min_lh_value = np.min(non_zero_loglh_array)
    max_lh_value = np.max(non_zero_loglh_array)

    factors = 0
    if max_lh_value > max_limit:
        factors = max_limit - max_lh_value - 1
    elif min_lh_value < min_limit:
        factors = min(min_limit - min_lh_value + 1, max_limit - max_lh_value - 1)
    loglikelihood_array += factors
    return factors


def optimize_likelihood_params(forest, character, observed_frequencies, model):
    """
    Optimizes the likelihood parameters (state frequencies and scaling factor) for the given trees.

    :param model: model of character evolution
    :type model: pastml.model.Model
    :param avg_br_len: avg branch length
    :type avg_br_len: float
    :param forest: trees of interest
    :type forest: list(ete3.Tree)
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
    :param tau: a smoothing factor to apply to branch lengths during likelihood calculation.
        If set to zero (default), zero internal branches will be collapsed instead.
    :type tau: float
    :return: optimized parameters and log likelihood: ((frequencies, scaling_factor), optimum)
    :rtype: tuple
    """
    bounds = model.get_bounds()

    def get_v(ps):
        if np.any(pd.isnull(ps)):
            return np.nan
        model.set_params_from_optimised(ps)
        res = sum(get_bottom_up_loglikelihood(tree=tree, character=character, is_marginal=True, model=model, alter=True)
                  for tree in forest)
        return np.inf if pd.isnull(res) else -res

    if np.any(observed_frequencies <= 0):
        observed_frequencies = np.maximum(observed_frequencies, 1e-10)

    x0_JC = model.get_optimised_parameters()
    optimise_frequencies = isinstance(model, ModelWithFrequencies) and model._optimise_frequencies
    x0_EFT = x0_JC
    if optimise_frequencies:
        model.frequencies = observed_frequencies
        x0_EFT = model.get_optimised_parameters()
    log_lh_JC = -get_v(x0_JC)
    log_lh_EFT = log_lh_JC if not optimise_frequencies else -get_v(x0_EFT)

    best_log_lh = max(log_lh_JC, log_lh_EFT)

    for i in range(100):
        if i == 0:
            vs = x0_JC
        elif optimise_frequencies and i == 1:
            vs = x0_EFT
        else:
            vs = np.random.uniform(bounds[:, 0], bounds[:, 1])
        fres = minimize(get_v, x0=vs, method='L-BFGS-B', bounds=bounds)
        if fres.success and not np.any(np.isnan(fres.x)):
            if -fres.fun >= best_log_lh:
                model.set_params_from_optimised(fres.x)
                return -fres.fun
    model.set_params_from_optimised(x0_JC if log_lh_JC >= log_lh_EFT else x0_EFT)
    return best_log_lh


def calculate_top_down_likelihood(tree, character, model):
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
    L_top_down(N1, i) = sum_j P(i -> j, dist(N1, P)) * L_top_down(P) * sum_k P(j -> k, dist(N2, P)) * L_bottom_up (N2).

    For the root node we assume its top-down likelihood to be 1 for all the states.

    :param model: model of character evolution
    :type model: pastml.models.Model
    :param character: character whose ancestral state likelihood is being calculated
    :type character: str
    :param tree: tree of interest (with bottom-up likelihood pre-calculated)
    :type tree: ete3.Tree
    :return: void, stores the node top-down likelihoods in the get_personalised_feature_name(feature, TD_LH) feature.
    """
    td_lh_feature = get_personalized_feature_name(character, TD_LH)
    td_lh_sf_feature = get_personalized_feature_name(character, TD_LH_SF)
    bu_lh_feature = get_personalized_feature_name(character, BU_LH)
    bu_lh_sf_feature = get_personalized_feature_name(character, BU_LH_SF)

    get_pij = model.get_Pij_t
    for node in tree.traverse('preorder'):
        calc_node_td_likelihood(node, td_lh_feature, td_lh_sf_feature, bu_lh_feature, bu_lh_sf_feature, get_pij)


def calc_node_td_likelihood(node, td_lh_feature, td_lh_sf_feature, bu_lh_feature, bu_lh_sf_feature, get_pij):
    if node.is_root():
        node.add_feature(td_lh_feature, np.ones(len(getattr(node, bu_lh_feature)), np.float64))
        node.add_feature(td_lh_sf_feature, 0)
        return
    parent = node.up
    node_pjis = np.transpose(get_pij(node.dist))
    node_contribution = getattr(node, bu_lh_feature).dot(node_pjis)
    node_contribution[node_contribution <= 0] = 1
    parent_loglikelihood = np.log10(getattr(parent, td_lh_feature)) \
                           + np.log10(getattr(parent, bu_lh_feature)) - np.log10(node_contribution)
    factors = getattr(parent, td_lh_sf_feature) \
              + getattr(parent, bu_lh_sf_feature) - getattr(node, bu_lh_sf_feature)
    factors += rescale_log(parent_loglikelihood)
    parent_likelihood = np.power(10, parent_loglikelihood)
    td_likelihood = parent_likelihood.dot(node_pjis)
    node.add_feature(td_lh_feature, np.maximum(td_likelihood, 0))
    node.add_feature(td_lh_sf_feature, factors)


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
    n = len(states)
    state2index = dict(zip(states, range(n)))

    for node in tree.traverse():
        node_states = getattr(node, feature, set())
        if not node_states:
            allowed_states = np.ones(n, dtype=int)
        else:
            allowed_states = np.zeros(n, dtype=int)
            for state in node_states:
                allowed_states[state2index[state]] = 1
        node.add_feature(allowed_states_feature, allowed_states)


def get_zero_clusters_with_states(tree, feature):
    """
    Returns the zero-distance clusters in the given tree.

    :param tree: ete3.Tree, the tree of interest
    :return: iterator of lists of nodes that are at zero distance from each other and have states specified for them.
    """

    def has_state(_):
        state = getattr(_, feature, None)
        return state is not None and state != ''

    todo = [tree]

    while todo:
        zero_cluster_with_states = []
        extension = [todo.pop()]

        while extension:
            n = extension.pop()
            if has_state(n):
                zero_cluster_with_states.append(n)
            for c in n.children:
                if c.dist == 0:
                    extension.append(c)
                else:
                    todo.append(c)
        if len(zero_cluster_with_states) > 1:
            yield zero_cluster_with_states


def alter_zero_node_allowed_states(tree, feature):
    """
    Alters the bottom-up likelihood arrays for zero-distance nodes
    to make sure they do not contradict with other zero-distance node siblings/ancestors/descendants.

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the likelihood is altered
    :return: void, modifies the get_personalised_feature_name(feature, BU_LH) feature to zero-distance nodes.
    """
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    allowed_state_feature_unaltered = get_personalized_feature_name(feature, ALLOWED_STATES + '.initial')

    altered_nodes = []

    for zero_cluster_with_states in get_zero_clusters_with_states(tree, feature):
        # If there is a common state do nothing
        counts = None
        for c in zero_cluster_with_states:
            if counts is None:
                counts = getattr(c, allowed_state_feature).copy()
            else:
                counts += getattr(c, allowed_state_feature)
        if counts.max() == len(zero_cluster_with_states):
            continue
        # Otherwise set all zero-cluster node states to state union
        allowed_states = None
        for c in zero_cluster_with_states:
            initial_allowed_states = getattr(c, allowed_state_feature).copy()
            if allowed_states is None:
                allowed_states = initial_allowed_states.copy()
            else:
                allowed_states[np.nonzero(initial_allowed_states)] = 1
            c.add_feature(allowed_state_feature, allowed_states)
            c.add_feature(allowed_state_feature_unaltered, initial_allowed_states)
            altered_nodes.append(c)
    return altered_nodes


def unalter_zero_node_allowed_states(altered_nodes, feature):
    """
    Unalters the bottom-up likelihood arrays for zero-distance nodes
    to contain ones only in their states.

    :param altered_nodes: list of modified nodes
    :param feature: str, character for which the likelihood was altered
    :return: void, modifies the get_personalised_feature_name(feature, BU_LH) feature to zero-distance nodes.
    """
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    allowed_state_feature_unaltered = get_personalized_feature_name(feature, ALLOWED_STATES + '.initial')
    for n in altered_nodes:
        initial_allowed_states = getattr(n, allowed_state_feature_unaltered)
        allowed_states = getattr(n, allowed_state_feature) & initial_allowed_states
        n.add_feature(allowed_state_feature,
                      (allowed_states if np.any(allowed_states > 0) else initial_allowed_states))


def unalter_zero_node_joint_states(altered_nodes, feature):
    """
    Unalters the joint states for zero-distance nodes
    to contain only their states.

    :param altered_nodes: list of modified nodes
    :param feature: str, character for which the likelihood was altered
    :return: void, modifies the get_personalised_feature_name(feature, BU_LH_JOINT_STATES) feature to zero-distance nodes.
    """
    lh_joint_state_feature = get_personalized_feature_name(feature, BU_LH_JOINT_STATES)
    allowed_state_feature_unaltered = get_personalized_feature_name(feature, ALLOWED_STATES + '.initial')
    for n in altered_nodes:
        initial_allowed_states = getattr(n, allowed_state_feature_unaltered)
        allowed_index = np.argmax(initial_allowed_states)
        if len(initial_allowed_states[initial_allowed_states > 0]) == 1:
            n.add_feature(lh_joint_state_feature, np.ones(len(initial_allowed_states), int) * allowed_index)
        else:
            joint_states = getattr(n, lh_joint_state_feature)
            for i in range(len(initial_allowed_states)):
                if not initial_allowed_states[joint_states[i]]:
                    joint_states[i] = allowed_index


def calculate_marginal_likelihoods(tree, feature, frequencies, clean_up=True):
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
        calc_node_marginal_likelihood(node, lh_feature, lh_sf_feature, bu_lh_feature, bu_lh_sf_feature, td_lh_feature,
                                      td_lh_sf_feature, allowed_state_feature, frequencies, clean_up)


def calc_node_marginal_likelihood(node, lh_feature, lh_sf_feature, bu_lh_feature, bu_lh_sf_feature, td_lh_feature,
                                  td_lh_sf_feature, allowed_state_feature, frequencies, clean_up):
    loglikelihood = np.log10(getattr(node, bu_lh_feature)) + np.log10(getattr(node, td_lh_feature)) \
                    + np.log10(frequencies * getattr(node, allowed_state_feature))
    factors = rescale_log(loglikelihood)
    node.add_feature(lh_feature, np.power(10, loglikelihood))
    node.add_feature(lh_sf_feature, factors + getattr(node, td_lh_sf_feature) + getattr(node, bu_lh_sf_feature))
    if clean_up:
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
        if not node.is_root():
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
        if hasattr(node, allowed_state_feature + '.initial'):
            marginal_likelihoods *= getattr(node, allowed_state_feature + '.initial')
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
            allowed_states = np.zeros(len(states), dtype=int)
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
        if hasattr(node, allowed_state_feature + '.initial'):
            marginal_likelihoods *= getattr(node, allowed_state_feature + '.initial')
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
    all_ones = np.ones(n, int)
    state2array = {}
    for index, state in enumerate(states):
        allowed_state_array = np.zeros(n, int)
        allowed_state_array[index] = 1
        state2array[state if by_name else index] = allowed_state_array
    if by_name:
        state2array[None] = all_ones
        state2array[''] = all_ones
    return all_ones, state2array


def ml_acr(forest, character, prediction_method, model, observed_frequencies, force_joint=True):
    """
    Calculates ML states on the trees and stores them in the corresponding feature.

    :param prediction_method: MPPA (marginal approximation), MAP (max a posteriori), JOINT or ML
    :type prediction_method: str
    :param forest: trees of interest
    :type forest: list(ete3.Tree)
    :param character: character for which the ML states are reconstructed
    :type character: str
    :param model: character state change model
    :type model: pastml.models.Model
    :return: mapping between reconstruction parameters and values
    :rtype: dict
    """
    logger = logging.getLogger('pastml')
    likelihood = \
        optimise_likelihood(forest=forest, character=character, model=model,
                            observed_frequencies=observed_frequencies)
    result = {LOG_LIKELIHOOD: likelihood, CHARACTER: character, METHOD: prediction_method, MODEL: model,
              STATES: model.states}

    results = []

    def process_reconstructed_states(method):
        if method == prediction_method or is_meta_ml(prediction_method):
            method_character = get_personalized_feature_name(character, method) \
                if prediction_method != method else character
            for tree in forest:
                convert_allowed_states2feature(tree, character, model.states, method_character)
            res = result.copy()
            res[CHARACTER] = method_character
            res[METHOD] = method
            results.append(res)

    def process_restricted_likelihood_and_states(method):
        restricted_likelihood = \
            sum(get_bottom_up_loglikelihood(tree=tree, character=character,
                                            is_marginal=True, model=model, alter=True) for tree in forest)
        note_restricted_likelihood(method, restricted_likelihood)
        process_reconstructed_states(method)

    def note_restricted_likelihood(method, restricted_likelihood):
        logger.debug('Log likelihood for {} after {} state selection:\t{:.6f}'
                     .format(character, method, restricted_likelihood))
        result[RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(method)] = restricted_likelihood

    if prediction_method != MAP:
        # Calculate joint restricted likelihood
        restricted_likelihood = \
            sum(get_bottom_up_loglikelihood(tree=tree, character=character,
                                            is_marginal=False, model=model, alter=True) for tree in forest)
        note_restricted_likelihood(JOINT, restricted_likelihood)
        for tree in forest:
            choose_ancestral_states_joint(tree, character, model.states, model.frequencies)
        process_reconstructed_states(JOINT)

    if is_marginal(prediction_method):
        mps = []
        for tree in forest:
            initialize_allowed_states(tree, character, model.states)
            altered_nodes = []
            if 0 == model.tau:
                altered_nodes = alter_zero_node_allowed_states(tree, character)
            get_bottom_up_loglikelihood(tree=tree, character=character, is_marginal=True, model=model, alter=False)
            calculate_top_down_likelihood(tree, character, model=model)
            calculate_marginal_likelihoods(tree, character, model.frequencies)
            # check_marginal_likelihoods(tree, character)
            mps.append(convert_likelihoods_to_probabilities(tree, character, model.states))

            if altered_nodes:
                unalter_zero_node_allowed_states(altered_nodes, character)
            choose_ancestral_states_map(tree, character, model.states)
        result[MARGINAL_PROBABILITIES] = pd.concat(mps, copy=False) if len(mps) != 1 else mps[0]
        process_restricted_likelihood_and_states(MAP)

        if MPPA == prediction_method or is_meta_ml(prediction_method):

            if ALL == prediction_method:
                pars_acr_results = parsimonious_acr(forest, character, MP, model.states,
                                                    model.forest_stats.num_nodes, model.forest_stats.num_tips)
                results.extend(pars_acr_results)
                for pars_acr_res in pars_acr_results:
                    for tree in forest:
                        _parsimonious_states2allowed_states(tree, pars_acr_res[CHARACTER], character, model.states)
                    try:
                        restricted_likelihood = \
                            sum(get_bottom_up_loglikelihood(tree=tree, character=character, is_marginal=True,
                                                            model=model, alter=True)
                                for tree in forest)
                        note_restricted_likelihood(pars_acr_res[METHOD], restricted_likelihood)
                    except PastMLLikelihoodError as e:
                        logger.error('{}\n{} parsimonious state selection is inconsistent in terms of ML.'
                                     .format(e.message, pars_acr_res[METHOD]))

            result[NUM_SCENARIOS], result[NUM_UNRESOLVED_NODES], result[NUM_STATES_PER_NODE] = 1, 0, 0
            for tree in forest:
                ns, nun, nspn = choose_ancestral_states_mppa(tree, character, model.states, force_joint=force_joint)
                result[NUM_SCENARIOS] *= ns
                result[NUM_UNRESOLVED_NODES] += nun
                result[NUM_STATES_PER_NODE] += nspn
            result[NUM_STATES_PER_NODE] /= model.forest_stats.num_nodes
            result[PERC_UNRESOLVED] = result[NUM_UNRESOLVED_NODES] * 100 / model.forest_stats.num_nodes
            logger.debug('{} node{} unresolved ({:.2f}%) for {} by {}, '
                         'i.e. {:.4f} state{} per node in average.'
                         .format(result[NUM_UNRESOLVED_NODES], 's are' if result[NUM_UNRESOLVED_NODES] != 1 else ' is',
                                 result[PERC_UNRESOLVED], character, MPPA,
                                 result[NUM_STATES_PER_NODE], 's' if result[NUM_STATES_PER_NODE] > 1 else ''))
            process_restricted_likelihood_and_states(MPPA)

    return results


def marginal_counts(forest, character, model, n_repetitions=1_000):
    """
    Calculates ML states on the trees and stores them in the corresponding feature.

    :param n_repetitions:
    :param forest: trees of interest
    :type forest: list(ete3.Tree)
    :param character: character for which the ML states are reconstructed
    :type character: str
    :param model: evolutionary model, F81 (Felsenstein 81-like), JC (Jukes-Cantor-like) or EFT (estimate from tips)
    :type model: Model
    :return: mapping between reconstruction parameters and values
    :rtype: dict
    """

    lh_feature = get_personalized_feature_name(character, LH)
    lh_sf_feature = get_personalized_feature_name(character, LH_SF)
    td_lh_feature = get_personalized_feature_name(character, TD_LH)
    td_lh_sf_feature = get_personalized_feature_name(character, TD_LH_SF)
    bu_lh_feature = get_personalized_feature_name(character, BU_LH)
    bu_lh_sf_feature = get_personalized_feature_name(character, BU_LH_SF)

    allowed_state_feature = get_personalized_feature_name(character, ALLOWED_STATES)
    initial_allowed_state_feature = allowed_state_feature + '.initial'
    state_count_feature = get_personalized_feature_name(character, STATE_COUNTS)

    get_pij = model.get_Pij_t
    n_states = len(model.states)
    state_ids = np.array(list(range(n_states)))

    result = np.zeros((n_states, n_states), dtype=float)

    for tree in forest:
        initialize_allowed_states(tree, character, model.states)
        altered_nodes = []
        if 0 == model.tau:
            altered_nodes = alter_zero_node_allowed_states(tree, character)
        get_bottom_up_loglikelihood(tree=tree, character=character, is_marginal=True, model=model, alter=False)
        calculate_top_down_likelihood(tree, character, model=model)

        for parent in tree.traverse('levelorder'):
            if parent.is_root():
                calc_node_marginal_likelihood(parent, lh_feature, lh_sf_feature, bu_lh_feature, bu_lh_sf_feature,
                                              td_lh_feature, td_lh_sf_feature, allowed_state_feature, model.frequencies,
                                              False)
                marginal_likelihoods = getattr(parent, lh_feature)
                marginal_probs = marginal_likelihoods / marginal_likelihoods.sum()
                # draw random states according to marginal probabilities (n_repetitions times)
                drawn_state_nums = Counter(np.random.choice(state_ids, size=n_repetitions, p=marginal_probs))
                parent_state_counts = np.array([drawn_state_nums[_] for _ in state_ids])
                parent.add_feature(state_count_feature, parent_state_counts)
            else:
                parent_state_counts = getattr(parent, state_count_feature)

            if parent in altered_nodes:
                initial_allowed_states = getattr(parent, initial_allowed_state_feature)
                ps_counts_initial = parent_state_counts * initial_allowed_states
                if np.count_nonzero(ps_counts_initial):
                    ps_counts_initial = n_repetitions * ps_counts_initial / ps_counts_initial.sum()
                else:
                    ps_counts_initial = n_repetitions * initial_allowed_states / initial_allowed_states.sum()
            else:
                ps_counts_initial = parent_state_counts

            same_state_counts = np.zeros(n_states)

            for node in parent.children:
                node_pjis = np.transpose(get_pij(node.dist))
                marginal_loglikelihood = np.log10(getattr(node, bu_lh_feature)) + np.log10(node_pjis) \
                                         + np.log10(model.frequencies * getattr(node, allowed_state_feature))
                rescale_log(marginal_loglikelihood)
                marginal_likelihood = np.power(10, marginal_loglikelihood)
                marginal_probs = marginal_likelihood / marginal_likelihood.sum(axis=1)[:, np.newaxis]
                state_counts = np.zeros(n_states, dtype=int)

                update_results = parent not in altered_nodes and node not in altered_nodes

                for j in range(n_states):
                    parent_count_j = parent_state_counts[j]
                    if parent_count_j:
                        drawn_state_nums_j = \
                            Counter(np.random.choice(state_ids, size=parent_count_j, p=marginal_probs[j, :]))
                        state_counts_j = np.array([drawn_state_nums_j[_] for _ in state_ids])
                        state_counts += state_counts_j
                        if update_results:
                            result[j, :] += state_counts_j
                            same_state_counts[j] += state_counts_j[j]

                if node in altered_nodes:
                    initial_allowed_states = getattr(node, initial_allowed_state_feature)
                    counts_initial = state_counts * initial_allowed_states
                    if np.count_nonzero(counts_initial):
                        counts_initial = n_repetitions * counts_initial / counts_initial.sum()
                    else:
                        counts_initial = n_repetitions * initial_allowed_states / initial_allowed_states.sum()
                else:
                    counts_initial = state_counts

                if not update_results:
                    norm_counts = counts_initial / counts_initial.sum()
                    for i in np.argwhere(ps_counts_initial > 0):
                        child_counts_adjusted = norm_counts * ps_counts_initial[i]
                        result[i, :] += child_counts_adjusted
                        same_state_counts[i] += child_counts_adjusted[i]

                node.add_feature(state_count_feature, state_counts)

            for i in range(n_states):
                result[i, i] -= np.minimum(ps_counts_initial[i], same_state_counts[i])

    return result / n_repetitions


def optimise_likelihood(forest, character, model, observed_frequencies):
    for tree in forest:
        initialize_allowed_states(tree, character, model.states)
    logger = logging.getLogger('pastml')
    likelihood = sum(get_bottom_up_loglikelihood(tree=tree, character=character,
                                                 is_marginal=True, model=model, alter=True)
                     for tree in forest)
    if np.isnan(likelihood):
        raise PastMLLikelihoodError('Failed to calculate the likelihood for your tree, '
                                    'please check that you do not have contradicting {} states specified '
                                    'for internal tree nodes, '
                                    'and if not - submit a bug at https://github.com/evolbioinfo/pastml/issues'
                                    .format(character))
    if not model.get_num_params():
        logger.debug('All the parameters are fixed for {}:\n{}{}.'
                     .format(character,
                             model._print_parameters(),
                             '\tlog likelihood:\t{:.6f}'.format(likelihood))
                     )
    else:
        logger.debug('Initial values for {} parameter optimisation:\n{}{}.'
                     .format(character,
                             model._print_parameters(),
                             '\tlog likelihood:\t{:.6f}'.format(likelihood))
                     )
        if not model.basic_params_fixed():
            model.fix_extra_params()
            likelihood = optimize_likelihood_params(forest=forest, character=character, model=model,
                                                    observed_frequencies=observed_frequencies)
            if np.any(np.isnan(likelihood) or likelihood == -np.inf):
                raise PastMLLikelihoodError('Failed to optimise the likelihood for your tree, '
                                            'please check that you do not have contradicting {} states specified '
                                            'for internal tree nodes, '
                                            'and if not - submit a bug at https://github.com/evolbioinfo/pastml/issues'
                                            .format(character))
            model.unfix_extra_params()
            if not model.extra_params_fixed():
                logger.debug('Pre-optimised basic parameters for {}:\n{}{}.'
                             .format(character,
                                     model._print_basic_parameters(),
                                     '\tlog likelihood:\t{:.6f}'.format(likelihood)))
        if not model.extra_params_fixed():
            likelihood = \
                optimize_likelihood_params(forest=forest, character=character, model=model,
                                           observed_frequencies=observed_frequencies)
            if np.any(np.isnan(likelihood) or likelihood == -np.inf):
                raise PastMLLikelihoodError('Failed to calculate the likelihood for your tree, '
                                            'please check that you do not have contradicting {} states specified '
                                            'for internal tree nodes, '
                                            'and if not - submit a bug at https://github.com/evolbioinfo/pastml/issues'
                                            .format(character))
        logger.debug('Optimised parameters for {}:\n{}{}'
                     .format(character,
                             model._print_parameters(),
                             '\tlog likelihood:\t{:.6f}'.format(likelihood)))
    return likelihood


def convert_allowed_states2feature(tree, feature, states, out_feature=None):
    if out_feature is None:
        out_feature = feature
    allowed_states_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    for node in tree.traverse():
        node.add_feature(out_feature, set(states[getattr(node, allowed_states_feature).astype(bool)]))


def _parsimonious_states2allowed_states(tree, ps_feature, feature, states):
    n = len(states)
    state2index = dict(zip(states, range(n)))
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    for node in tree.traverse():
        pars_states = getattr(node, ps_feature)
        allowed_states = np.zeros(n, dtype=int)
        for state in pars_states:
            allowed_states[state2index[state]] = 1
        node.add_feature(allowed_state_feature, allowed_states)


class PastMLLikelihoodError(Exception):

    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        if self.message:
            return 'PastMLLikelihoodError, {}'.format(self.message)
        else:
            return 'PastMLLikelihoodError has been raised.'
