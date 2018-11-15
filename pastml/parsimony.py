import logging
from collections import namedtuple

import numpy as np

from pastml import get_personalized_feature_name, ALLOWED_STATES

DOWNPASS = 'downpass'
ACCTRAN = 'AccTran'
DELTRAN = 'DelTran'

BU_PARS_STATES = 'BOTTOM_UP_PARSIMONY'
TD_PARS_STATES = 'TOP_DOWN_PARSIMONY'
PARS_STATES = 'PARSIMONY'
PARS_STATE2NUM = 'PARSIMONY_STEPS'


def is_parsimonious(method):
    """
    Checks if the method is max likelihood, i.e. is either joint or one of the marginal ones
    (marginal itself, or MAP, or MPPA).
    :param method: str, the ancestral state prediction method used by PASTML.
    :return: bool
    """
    return method in {DOWNPASS, ACCTRAN, DELTRAN}


def initialise_tip_parsimonious_states(tree, feature, states):
    """
    Initializes the bottom-up state arrays for tips based on their states given by the feature.

    :param tree: ete3.Tree, tree for which the tip states are to be initialized
    :param feature: str, feature in which the tip states are stored (the value could be None for a missing state)
    :param states: numpy array, possible states.
    :return: void, adds the get_personalised_feature_name(feature, BU_PARS) feature to tree tips.
    """
    ps_feature = get_personalized_feature_name(feature, BU_PARS_STATES)
    all_states = set(states)

    state2array = {state: {state} for state in states}
    state2array[None] = all_states
    state2array[''] = all_states

    for tip in tree:
        tip.add_feature(ps_feature, state2array[getattr(tip, feature, '')])


def get_most_common_states(state_list):
    """
    Gets the set of most common states among the state sets contained in the iterable argument
    :param state_list: list of allowed state arrays (that have 1 if the corresponding state is allowed else 0)
    :return: set of most common states
    """
    allowed_states = np.sum(state_list, axis=0)
    most_common_count = allowed_states.max()
    allowed_states[allowed_states < most_common_count] = 0
    allowed_states[allowed_states == most_common_count] = 1
    return allowed_states


def uppass(tree, feature):
    """
    UPPASS traverses the tree starting from the tips and going up till the root,
    and assigns to each parent node a state based on the states of its child nodes.

    if N is a tip:
    S(N) <- state of N
    else:
      L, R <- left and right children of N
      UPPASS(L)
      UPPASS(R)
      if S(L) intersects with S(R):
         S(N) <- intersection(S(L), S(R))
      else:
         S(N) <- union(S(L), S(R))

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the parsimonious states are reconstructed
    :return: void, adds get_personalized_feature_name(feature, BU_PARS_STATES) feature to the tree nodes
    """

    ps_feature = get_personalized_feature_name(feature, BU_PARS_STATES)
    allowed_states_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.add_feature(ps_feature, getattr(node, allowed_states_feature))
        else:
            node.add_feature(ps_feature, get_most_common_states(getattr(child, ps_feature) for child in node.children))


def acctran(tree, feature):
    """
    ACCTRAN (accelerated transformation) (Farris, 1970) aims at reducing the number of ambiguities
    in the parsimonious result. ACCTRAN forces the state changes to be performed as close to the root as possible,
    and therefore prioritises the reverse mutations.

    if N is not a tip:
        L, R <- left and right children of N
        if intersection(S(N), S(L)) is not empty:
            S(L) <- intersection(S(N), S(L))
        if intersection(S(N), S(R)) is not empty:
            S(R) <- intersection(S(N), S(R))
        ACCTRAN(L)
        ACCTRAN(R)

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the parsimonious states are reconstructed
    :return: void, adds get_personalized_feature_name(feature, ALLOWED_STATES) feature to the tree nodes
    """

    ps_feature_down = get_personalized_feature_name(feature, BU_PARS_STATES)
    ps_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    for node in tree.traverse('preorder'):
        if node.is_root():
            node.add_feature(ps_feature, getattr(node, ps_feature_down))
        node_states = getattr(node, ps_feature)
        for child in node.children:
            child_states = getattr(child, ps_feature_down)
            state_intersection = node_states & child_states
            child.add_feature(ps_feature, state_intersection if np.any(state_intersection) else child_states)
        node.del_feature(ps_feature_down)


def downpass(tree, feature, states):
    """
    DOWNPASS traverses the tree starting from the root and going down till the tips,
    and for each node combines the state information from its supertree and its subtree (calculated at UPPASS).
    As the root state was already the most parsimonious after the UPPASS,
    we skip it and start directly with the root children.

    if N is not a tip:
        L, R <- left and right children of N
        if N is root:
            UP_S(N) <- union of all states
        else:
            P <- parent of N
            B <- brother of N
            UP_S(N) <- most_common_states(UP_S(P), S(B))
        S(N) <- most_common_states(UP_S(N), S(L), S(R))
        DOWNPASS(L)
        DOWNPASS(R)

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the parsimonious states are reconstructed
    :return: void, adds get_personalized_feature_name(feature, ALLOWED_STATES) feature to the tree nodes
    """

    ps_feature_down = get_personalized_feature_name(feature, BU_PARS_STATES)
    ps_feature_up = get_personalized_feature_name(feature, TD_PARS_STATES)
    ps_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    for node in tree.traverse('preorder'):
        if node.is_root():
            node.add_feature(ps_feature_up, np.ones(len(states), dtype=np.int))
        else:
            node.add_feature(ps_feature_up,
                             get_most_common_states([getattr(node.up, ps_feature_up)]
                                                    + [getattr(sibling, ps_feature_down) for sibling in node.up.children
                                                       if sibling != node]))
        if not node.is_leaf():
            node.add_feature(ps_feature,
                             get_most_common_states([getattr(node, ps_feature_up)]
                                                    + [getattr(child, ps_feature_down) for child in node.children]))
        # try to resolve unresolved tips using the up information if possible
        elif np.sum(getattr(node, ps_feature_down)) > 1:
            node.add_feature(ps_feature,
                             get_most_common_states([getattr(node, ps_feature_up), getattr(node, ps_feature_down)]))
        else:
            node.add_feature(ps_feature, getattr(node, ps_feature_down))

    for node in tree.traverse():
        node.del_feature(ps_feature_down)
        node.del_feature(ps_feature_up)


def deltran(tree, feature):
    """
    DELTRAN (delayed transformation) (Swofford & Maddison, 1987) aims at reducing the number of ambiguities
    in the parsimonious result. DELTRAN makes the changes as close as possible to the leaves,
    hence prioritizing parallel mutations. DELTRAN is performed after DOWNPASS.

    if N is not a root:
        P <- parent(N)
        if intersection(S(N), S(P)) is not empty:
            S(N) <- intersection(S(N), S(P))
    if N is not a tip:
        L, R <- left and right children of N
        DELTRAN(L)
        DELTRAN(R)

    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the parsimonious states are reconstructed
    :return: void, modifies get_personalized_feature_name(feature, PARS_STATES) feature of the tree nodes
    """
    ps_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

    for node in tree.traverse('preorder'):
        if not node.is_root():
            node_states = getattr(node, ps_feature)
            parent_states = getattr(node.up, ps_feature)
            state_intersection = node_states & parent_states
            if np.any(state_intersection):
                node.add_feature(ps_feature, state_intersection)


ACRParsimoniousResult = namedtuple('ACRParsimoniousResult',
                                   field_names=['steps', 'character', 'states', 'method'])


def parsimonious_acr(tree, feature, prediction_method, states):
    """
    Calculates parsimonious states on the tree and stores them in the corresponding feature.

    :param states: numpy array of possible states
    :param prediction_method: str, ACCTRAN (accelerated transformation), DELTRAN (delayed transformation) or DOWNPASS
    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the parsimonious states are reconstructed
    :return: void, add parsimonious states as the `feature` feature to each node
    """
    uppass(tree, feature)
    if ACCTRAN == prediction_method:
        acctran(tree, feature)
    else:
        downpass(tree, feature, states)
        if DELTRAN == prediction_method:
            deltran(tree, feature)
    num_steps = get_num_parsimonious_steps(tree, feature)
    logging.info("Parsimonious reconstruction for {} requires {} state changes.\n".format(feature, num_steps))
    return ACRParsimoniousResult(steps=num_steps, character=feature, states=states, method=prediction_method)


def get_num_parsimonious_steps(tree, feature):
    ps_feature = get_personalized_feature_name(feature, ALLOWED_STATES)
    ps_feature_num = get_personalized_feature_name(feature, PARS_STATE2NUM)

    for node in tree.traverse('postorder'):
        states = getattr(node, ps_feature)
        nums = states.copy()
        # number of changes cannot be larger than the number of branches in the tree
        nums[states == 0] = 2 * len(tree)
        nums[states == 1] = 0
        node.add_feature(ps_feature_num, nums)
        for child in node.children:
            child_nums = getattr(child, ps_feature_num)
            nums += np.min([child_nums, np.ones(len(child_nums), dtype=np.int) * child_nums.min() + 1], axis=0)
            child.del_feature(ps_feature_num)
    state2num = getattr(tree, ps_feature_num)
    tree.del_feature(ps_feature_num)
    return state2num.min()
