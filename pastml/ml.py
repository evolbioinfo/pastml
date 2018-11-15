import logging
from collections import namedtuple, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pastml import get_personalized_feature_name

ACRMLJointResult = namedtuple('ACRMLJointResult',
                              field_names=['likelihood', 'restricted_likelihood', 'frequencies', 'sf', 'norm_sf',
                                           'character', 'states', 'method', 'model'])

ACRMLMarginalResult = namedtuple('ACRMLMarginalResult',
                                 field_names=['likelihood', 'restricted_likelihood', 'frequencies', 'sf', 'norm_sf',
                                              'character', 'states', 'method', 'model', 'marginal_probabilities'])

MIN_VALUE = np.log10(np.finfo(np.float64).eps)
MAX_VALUE = np.log10(np.finfo(np.float64).max)

JOINT = 'JOINT'
MPPA = 'MPPA'
MAP = 'MAP'

JC = 'JC'
F81 = 'F81'
EFT = 'EFT'

BU_LH = 'BOTTOM_UP_LIKELIHOOD'
TD_LH = 'TOP_DOWN_LIKELIHOOD'
LH = 'LIKELIHOOD'
LH_SF = 'LIKELIHOOD_SF'
BU_LH_SF = 'BOTTOM_UP_LIKELIHOOD_SF'
BU_LH_JOINT_STATES = 'BOTTOM_UP_LIKELIHOOD_JOINT_STATES'
TD_LH_SF = 'TOP_DOWM_LIKELIHOOD_SF'
ALLOWED_STATES = 'ALLOWED_STATES'


def is_marginal(method):
    """
    Checks if the method is marginal, i.e. is either marginal itself, or MAP, or MPPA.
    :param method: str, the ancestral state prediction method used by PASTML.
    :return: bool
    """
    return method in {MPPA, MAP}


def is_ml(method):
    """
    Checks if the method is max likelihood, i.e. is either joint or one of the marginal ones
    (marginal itself, or MAP, or MPPA).
    :param method: str, the ancestral state prediction method used by PASTML.
    :return: bool
    """
    return method == JOINT or is_marginal(method)


def get_mu(frequencies):
    """
    Calculates the mutation rate for F81 (and JC that is a simplification of it),
    as \mu = 1 / (1 - sum_i \pi_i^2). This way the overall rate of mutation -\mu trace(\Pi Q) is 1.
    See [Gascuel "Mathematics of Evolution and Phylogeny" 2005] for further details.

    :param frequencies: numpy array of state frequencies \pi_i
    :return: mutation rate \mu = 1 / (1 - sum_i \pi_i^2)
    """
    return 1. / (1. - frequencies.dot(frequencies))


def get_pij(frequencies, mu, t, sf):
    """
    Calculate the probability of substitution i->j over time t, given the mutation rate mu:
    For K81 (and JC which is a simpler version of it)
    Pij(t) = \pi_j (1 - exp(-mu t)) + exp(-mu t), if i == j, \pi_j (1 - exp(-mu t)), otherwise
    [Gascuel "Mathematics of Evolution and Phylogeny" 2005].

    :param frequencies: numpy array of state frequencies \pi_i
    :param mu: float, mutation rate: \mu = 1 / (1 - sum_i \pi_i^2)
    :param t: float, time t
    :param sf: float, scaling factor by which t should be multiplied.
    :return: numpy matrix Pij(t) = \pi_j (1 - exp(-mu t)) + exp(-mu t), if i == j, \pi_j (1 - exp(-mu t)), otherwise
    """
    # if mu == inf (e.g. just one state) and t == 0, we should prioritise mu
    exp_mu_t = 0. if (mu == np.inf) else np.exp(-mu * t * sf)
    return (1 - exp_mu_t) * frequencies + np.eye(len(frequencies)) * exp_mu_t


def get_bottom_up_likelihood(tree, feature, frequencies, sf, is_marginal=True):
    """
    Calculates the bottom-up likelihood for the given tree.
    The likelihood for each node is stored in the corresponding feature,
    given by get_personalised_feature_name(feature, BU_LH).

    :param is_marginal: bool, whether the likelihood reconstruction is marginal (true) or joint (false)
    :param tree: ete3.Tree tree
    :param feature: str, character for which the likelihood is calculated
    :param frequencies: numpy array of state frequencies \pi_i
    :param sf: float, scaling factor
    :return: float, the log likelihood
    """
    lh_sf_feature = get_personalized_feature_name(feature, BU_LH_SF)
    lh_feature = get_personalized_feature_name(feature, BU_LH)
    lh_joint_state_feature = get_personalized_feature_name(feature, BU_LH_JOINT_STATES)
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    mu = get_mu(frequencies)
    for node in tree.traverse('postorder'):
        likelihood_array = np.ones(len(frequencies), dtype=np.float64)

        for child in node.children:
            child_likelihoods = get_pij(frequencies, mu, child.dist, sf) * getattr(child, lh_feature)

            if is_marginal:
                likelihood_array *= child_likelihoods.sum(axis=1)
            else:
                likelihood_array *= child_likelihoods.max(axis=1)
                child_states = child_likelihoods.argmax(axis=1)
                child.add_feature(lh_joint_state_feature, child_states)

        if np.all(likelihood_array == 0):
            return -np.inf

        factors = rescale(likelihood_array, node, going_up=True)
        likelihood_array *= getattr(node, allowed_state_feature)
        node.add_feature(lh_feature, likelihood_array)
        node.add_feature(lh_sf_feature, factors + sum(getattr(_, lh_sf_feature) for _ in node.children))
    return np.log(getattr(tree, lh_feature).dot(frequencies)) - getattr(tree, lh_sf_feature) * np.log(10)


def rescale(likelihood_array, node, going_up=True):
    """
    Rescales the likelihood array if it gets too small/large, by multiplying it by a factor of 10.
    :param likelihood_array: numpy array containing the likelihood to be rescaled
    :param node: ete3.TreeNode who's likelihood is about to be rescaled.
    :param going_up: bool, whether we are going bottom-up (true) or top-down (false) in the tree with this likelihood calculation.
    :return: float, factor of 10 by which the likelihood array has been multiplies.
    """

    min_lh_value = np.log10(np.min(likelihood_array[np.nonzero(likelihood_array)]))
    max_lh_value = np.log10(np.max(likelihood_array[np.nonzero(likelihood_array)]))

    num_siblings = 2
    # if we go up, this likelihood will be multiplied by those of our siblings at the next step
    if going_up and not node.is_root():
        num_siblings = len(node.up.children)
    # if we go down, this likelihood will be multiplied by those of our children at the next step
    elif not going_up and not node.is_leaf():
        num_siblings = len(node.children)

    factors = 0

    if max_lh_value > MAX_VALUE / num_siblings:
        factors = MAX_VALUE / num_siblings - max_lh_value
        likelihood_array *= np.power(10, factors)
    elif min_lh_value < MIN_VALUE / num_siblings:
        factors = min(-min_lh_value, MAX_VALUE / num_siblings - max_lh_value)
        likelihood_array *= np.power(10, factors)
    return factors


def optimize_likelihood_params(tree, feature, frequencies, sf, avg_br_len, optimise_sf=True, optimise_frequencies=True):
    """
    Optimizes the likelihood parameters (state frequencies and scaling factor) for the given tree.
    :param avg_br_len: float, avg branch length
    :param tree: ete3.Tree, tree of interest
    :param feature: str, character for which the likelihood is optimised
    :param frequencies: numpy array of initial state frequencies
    :param sf: float, initial scaling factor
    :param optimise_sf: bool, whether the scaling factor needs to be optimised
    :param optimise_frequencies: bool, whether the state frequencies need to be optimised
    :return: tuple (frequencies, scaling_factor) with the optimized parameters
    """
    bounds = []
    if optimise_frequencies:
        bounds += [np.array([1e-6, 10e6], np.float64)] * (len(frequencies) - 1)
    if optimise_sf:
        bounds += [np.array([0.001 / avg_br_len, 10. / avg_br_len])]
    bounds = np.array(bounds, np.float64)

    def get_freq_sf_from_params(ps):
        freqs = frequencies
        if optimise_frequencies:
            freqs = np.hstack((ps[: (len(frequencies) - 1)], [1.]))
            freqs /= freqs.sum()
        sf_val = ps[(len(frequencies) - 1) if optimise_frequencies else 0] if optimise_sf else sf
        return freqs, sf_val

    def get_v(ps):
        if np.any(pd.isnull(ps)):
            return np.nan
        freqs, sf_val = get_freq_sf_from_params(ps)
        res = get_bottom_up_likelihood(tree, feature, freqs, sf_val, True)
        return np.inf if pd.isnull(res) else -res

    params = None
    optimum = None
    for i in range(5 if optimise_frequencies else 3):
        if i == 0:
            vs = np.hstack((frequencies[:-1] / frequencies[-1] if optimise_frequencies else [],
                            [sf] if optimise_sf else []))
        elif i == 1 and optimise_frequencies:
            vs = np.hstack((np.ones(len(frequencies) - 1, np.float64),
                            [sf] if optimise_sf else []))
        else:
            vs = np.random.uniform(bounds[:, 0], bounds[:, 1])
        fres = minimize(get_v, x0=vs, method='L-BFGS-B', bounds=bounds)
        if fres.success and not np.any(np.isnan(fres.x)):
            logging.info('Calculated an optimum candidate of {}'.format(fres.fun))
            if optimum is None or fres.fun < optimum:
                params = fres.x
                optimum = fres.fun
    if params is None:
        return None, None
    return get_freq_sf_from_params(params)


def calculate_top_down_likelihood(tree, feature, frequencies, sf):
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

    :param tree: ete3.Tree, the tree of interest (with bottom-up likelihood precalculated)
    :param frequencies: numpy array of state frequencies
    :return: void, stores the node top-down likelihoods in the get_personalised_feature_name(feature, TD_LH) feature.
    """

    lh_feature = get_personalized_feature_name(feature, TD_LH)
    lh_sf_feature = get_personalized_feature_name(feature, TD_LH_SF)
    bu_lh_feature = get_personalized_feature_name(feature, BU_LH)
    bu_lh_sf_feature = get_personalized_feature_name(feature, BU_LH_SF)

    mu = get_mu(frequencies)

    for node in tree.traverse('preorder'):
        if node.is_root():
            node.add_feature(lh_feature, np.ones(len(frequencies), np.float64))
            node.add_feature(lh_sf_feature, 0)
            continue

        parent = node.up
        parent_bu_likelihood = getattr(parent, bu_lh_feature)

        node_pjis = np.transpose(get_pij(frequencies, mu, node.dist, sf))
        node_contribution = getattr(node, bu_lh_feature).dot(node_pjis)

        parent_likelihood = getattr(parent, lh_feature) * parent_bu_likelihood
        parent_likelihood[np.nonzero(parent_likelihood)] /= node_contribution[np.nonzero(parent_likelihood)]
        factors = getattr(parent, lh_sf_feature) + getattr(parent, bu_lh_sf_feature) - getattr(node, bu_lh_sf_feature)

        td_likelihood = parent_likelihood.dot(node_pjis)
        factors += rescale(td_likelihood, node, going_up=False)

        node.add_feature(lh_feature, td_likelihood)
        node.add_feature(lh_sf_feature, factors)


def initialize_allowed_states(tree, feature, states):
    """
    Initializes the allowed state arrays for tips based on their states given by the feature.
    :param tree: ete3.Tree, tree for which the tip likelihoods are to be initialized
    :param feature: str, feature in which the tip states are stored (the value could be None for a missing state)
    :param states: numpy array of ordered states.
    :return: void, adds the get_personalised_feature_name(feature, ALLOWED_STATES) feature to tree tips.
    """
    allowed_states_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    all_ones, state2array = get_state2allowed_states(states)

    for node in tree.traverse():
        if node.is_leaf():
            node.add_feature(allowed_states_feature, state2array[getattr(node, feature, '')])
        else:
            node.add_feature(allowed_states_feature, all_ones)


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
        state = getattr(tip, feature, None)
        if state is not None and state != '':
            allowed_states = np.zeros(len(state2index), np.int)
            allowed_states[state2index[state]] = 1
            tip.add_feature(allowed_state_feature, allowed_states)


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
        state = getattr(tip, feature, None)
        if state is not None and state != '':
            tip.add_feature(lh_joint_state_feature, np.ones(len(state2index), np.int) * state2index[state])


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


def choose_ancestral_states_mppa(tree, feature, states):
    """
    Chooses node ancestral states based on their marginal probabilities using MPPA method.

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the ancestral states are to be chosen
    :param states: numpy.array of possible character states in order corresponding to the probabilities array
    :return: void, add ancestral states as the `feature` feature to each node
    (as a list if multiple states are possible or as a string if only one state is chosen)
    """
    lh_feature = get_personalized_feature_name(feature, LH)
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    n = len(states)
    _, state2array = get_state2allowed_states(states, False)

    for node in tree.traverse():
        marginal_likelihoods = getattr(node, lh_feature)
        sorted_marginal_probs = np.sort(marginal_likelihoods / marginal_likelihoods.sum())
        best_k = n
        best_correstion = np.inf
        for k in range(1, n):
            correction = np.hstack((np.zeros(n - k), np.ones(k) / k)) - sorted_marginal_probs
            correction = correction.dot(correction)
            if correction < best_correstion:
                best_correstion = correction
                best_k = k

        indices_selected = sorted(range(n), key=lambda _: -marginal_likelihoods[_])[:best_k]
        if best_k == 1:
            allowed_states = state2array[indices_selected[0]]
        else:
            allowed_states = np.zeros(len(states), dtype=np.int)
            allowed_states[indices_selected] = 1
        node.add_feature(allowed_state_feature, allowed_states)


def choose_ancestral_states_map(tree, feature, states):
    """
    Chooses node ancestral states based on their marginal probabilities using MAP method.

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the ancestral states are to be chosen
    :param states: numpy.array of possible character states in order corresponding to the probabilities array
    :return: void, add ancestral states as the `feature` feature to each node
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
    :return: void, adds ancestral states as the `feature` feature to each node
    """
    lh_feature = get_personalized_feature_name(feature, BU_LH)
    lh_state_feature = get_personalized_feature_name(feature, BU_LH_JOINT_STATES)
    allowed_state_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    _, state2array = get_state2allowed_states(states, False)

    def chose_consistent_state(node, state_index):
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


def ml_acr(tree, feature, prediction_method, model, states, avg_br_len, freqs=None, sf=None):
    optimise_frequencies = F81 == model
    n = len(states)
    state2index = dict(zip(states, range(n)))
    missing_data = 0.
    if freqs is not None and F81 != model:
        logging.warning('Some frequencies were specified in the parameter file, but the selected model ({}) ignores them. '
                        'Use F81 for taking user-specified frequencies into account.'.format(model))
    if JC == model:
        frequencies = np.ones(n, dtype=np.float64) / n
    elif freqs is not None and F81 == model:
        frequencies = freqs
        optimise_frequencies = False
    else:
        frequencies = np.zeros(n, np.float64)
        for _ in tree:
            state = getattr(_, feature, None)
            if state is not None and state != '':
                frequencies[state2index[state]] += 1
            else:
                missing_data += 1
        total_count = frequencies.sum() + missing_data
        frequencies /= frequencies.sum()
        missing_data /= total_count

    initialize_allowed_states(tree, feature, states)
    alter_zero_tip_allowed_states(tree, feature)
    if sf:
        optimise_sf = False
    else:
        sf = 1. / avg_br_len
        optimise_sf = True
    likelihood = get_bottom_up_likelihood(tree, feature, frequencies, sf, True)
    logging.info('Initial {} values:{}{}{}{}.\n'
                 .format(feature,
                         ''.join('\n\tfrequency of {}:\t{:.3f}'.format(state, frequencies[state2index[state]])
                                 for state in states),
                         '\n\tfraction of missing data:\t{:.3f}'.format(missing_data) if missing_data else '',
                         '\n\tSF:\t{:.3f}, i.e. {:.3f} changes per avg branch'.format(sf, sf * avg_br_len),
                         '\n\tlog likelihood:\t{:.3f}'.format(likelihood)))
    if optimise_sf or optimise_frequencies:
        frequencies, sf = optimize_likelihood_params(tree, feature, frequencies, sf,
                                                     optimise_frequencies=optimise_frequencies, optimise_sf=optimise_sf,
                                                     avg_br_len=avg_br_len)
        likelihood = get_bottom_up_likelihood(tree, feature, frequencies, sf, True)
        logging.info('Optimised {} values:{}{}{}\n'
                     .format(feature,
                             ''.join('\n\tfrequency of {}:\t{:.3f}'.format(state, frequencies[state2index[state]])
                                     for state in states) if optimise_frequencies else '',
                             '\n\tSF:\t{:.3f}, i.e. {:.3f} changes per avg branch'.format(sf, sf * avg_br_len),
                             '\n\tlog likelihood:\t{:.3f}'.format(likelihood)))
    else:
        logging.info('Both scaling factor and frequencies are fixed for {}.\n'.format(feature))

    if is_marginal(prediction_method):
        calculate_top_down_likelihood(tree, feature, frequencies, sf)
        unalter_zero_tip_allowed_states(tree, feature, state2index)
        calculate_marginal_likelihoods(tree, feature, frequencies)
        marginal_df = convert_likelihoods_to_probabilities(tree, feature, states)
        if MPPA == prediction_method:
            logging.info('Choosing MPPA ancestral states for {}.\n'.format(feature))
            choose_ancestral_states_mppa(tree, feature, states)
        elif MAP == prediction_method:
            logging.info('Choosing MAP ancestral states for {}.\n'.format(feature))
            choose_ancestral_states_map(tree, feature, states)
        alter_zero_tip_allowed_states(tree, feature)
        restricted_likelihood = get_bottom_up_likelihood(tree, feature, frequencies, sf, True)
        unalter_zero_tip_allowed_states(tree, feature, state2index)
        logging.info('Log likelihood for {} after state selection:\t{:.3f}\n'.format(feature, restricted_likelihood))
        result = ACRMLMarginalResult(likelihood=likelihood, restricted_likelihood=restricted_likelihood,
                                   frequencies=frequencies, sf=sf, norm_sf=sf * avg_br_len,
                                   method=prediction_method, model=model, character=feature, states=states,
                                   marginal_probabilities=marginal_df)
    # joint
    else:
        get_bottom_up_likelihood(tree, feature, frequencies, sf, False)
        unalter_zero_tip_joint_states(tree, feature, state2index)
        logging.info('Choosing joint ancestral states for {}.\n'.format(feature))
        choose_ancestral_states_joint(tree, feature, states, frequencies)
        alter_zero_tip_allowed_states(tree, feature)
        restricted_likelihood = get_bottom_up_likelihood(tree, feature, frequencies, sf, False)
        unalter_zero_tip_allowed_states(tree, feature, state2index)
        logging.info('Log likelihood for {} after state selection:\t{:.3f}\n'.format(feature, restricted_likelihood))
        result = ACRMLJointResult(likelihood=likelihood, restricted_likelihood=restricted_likelihood,
                            frequencies=frequencies, sf=sf, norm_sf=sf * avg_br_len,
                            method=prediction_method, model=model, character=feature, states=states)

    allowed_states_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    for node in tree.traverse():
        selected_states = states[getattr(node, allowed_states_feature).astype(bool)].tolist()
        node.add_feature(feature, selected_states[0] if len(selected_states) == 1 else selected_states)

    return result
