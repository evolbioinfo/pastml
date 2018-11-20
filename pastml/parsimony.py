import logging
from collections import Counter

from pastml import get_personalized_feature_name, METHOD, STATES, CHARACTER, NUM_SCENARIOS, NUM_UNRESOLVED_NODES, \
    NUM_NODES, NUM_TIPS

STEPS = 'steps'

DOWNPASS = 'DOWNPASS'
ACCTRAN = 'ACCTRAN'
DELTRAN = 'DELTRAN'

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


def get_most_common_states(state_iterable):
    """
    Gets the set of most common states among the state sets contained in the iterable argument
    :param state_iterable: iterable of state sets
    :return: set of most common states
    """
    state_counter = Counter()
    for states in state_iterable:
        state_counter.update(states)
    max_count = state_counter.most_common(1)[0][1]
    return {state for (state, count) in state_counter.items() if count == max_count}


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

    for node in tree.traverse('postorder'):
        if not node.is_leaf():
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
    :return: void, adds get_personalized_feature_name(feature, PARS_STATES) feature to the tree nodes
    """

    ps_feature_down = get_personalized_feature_name(feature, BU_PARS_STATES)
    ps_feature = get_personalized_feature_name(feature, PARS_STATES)

    for node in tree.traverse('preorder'):
        if node.is_root():
            node.add_feature(ps_feature, getattr(node, ps_feature_down))
        node_states = getattr(node, ps_feature)
        for child in node.children:
            child_states = getattr(child, ps_feature_down)
            state_intersection = node_states & child_states
            child.add_feature(ps_feature, state_intersection if state_intersection else child_states)
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
    :return: void, adds get_personalized_feature_name(feature, PARS_STATES) feature to the tree nodes
    """

    ps_feature_down = get_personalized_feature_name(feature, BU_PARS_STATES)
    ps_feature_up = get_personalized_feature_name(feature, TD_PARS_STATES)
    ps_feature = get_personalized_feature_name(feature, PARS_STATES)

    for node in tree.traverse('preorder'):
        if node.is_root():
            node.add_feature(ps_feature_up, set(states))
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
        elif len(getattr(node, ps_feature_down)) > 1:
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
    ps_feature = get_personalized_feature_name(feature, PARS_STATES)

    for node in tree.traverse('preorder'):
        if not node.is_root():
            node_states = getattr(node, ps_feature)
            parent_states = getattr(node.up, ps_feature)
            state_intersection = node_states & parent_states
            if state_intersection:
                node.add_feature(ps_feature, state_intersection)


def parsimonious_acr(tree, feature, prediction_method, states, num_nodes, num_tips):
    """
    Calculates parsimonious states on the tree and stores them in the corresponding feature.

    :param states: numpy array of possible states
    :param prediction_method: str, ACCTRAN (accelerated transformation), DELTRAN (delayed transformation) or DOWNPASS
    :param tree: ete3.Tree, the tree of interest
    :param feature: str, character for which the parsimonious states are reconstructed
    :return: dict, mapping between reconstruction parameters and values
    """
    initialise_tip_parsimonious_states(tree, feature, states)
    uppass(tree, feature)
    if ACCTRAN == prediction_method:
        acctran(tree, feature)
    else:
        downpass(tree, feature, states)
        if DELTRAN == prediction_method:
            deltran(tree, feature)
    num_steps = get_num_parsimonious_steps(tree, feature)
    logger = logging.getLogger('pastml')
    logger.debug("Parsimonious reconstruction for {} requires {} state changes."
                 .format(feature, num_steps))
    num_scenarios, unresolved_nodes = choose_parsimonious_states(tree, feature)
    logger.debug('{} node{} unresolved ({:.2f}%) for {}, leading to {:g} ancestral scenario{}.'
                 .format(unresolved_nodes, 's are' if unresolved_nodes > 1 else ' is',
                         unresolved_nodes * 100 / num_nodes, feature,
                         num_scenarios, 's' if num_scenarios > 1 else ''))

    return {STEPS: num_steps, CHARACTER: feature, STATES: states, METHOD: prediction_method,
            NUM_SCENARIOS: num_scenarios, NUM_UNRESOLVED_NODES: unresolved_nodes,
            NUM_NODES: num_nodes, NUM_TIPS: num_tips}


def choose_parsimonious_states(tree, feature):
    """
    Converts the content of the get_personalized_feature_name(feature, PARS_STATES) node feature to the predicted states
    and stores them in the `feature` feature to each node.
    The get_personalized_feature_name(feature, PARS_STATES) is deleted.

    :param feature: str, character for which the parsimonious states are reconstructed
    :param tree: ete3.Tree, the tree of interest
    :return: int, number of ancestral scenarios selected,
    calculated by multiplying the number of selected states for all nodes.
    Also adds parsimonious states as the `feature` feature to each node
    """
    num_scenarios = 1
    unresolved_nodes = 0
    ps_feature = get_personalized_feature_name(feature, PARS_STATES)
    for node in tree.traverse():
        states = getattr(node, ps_feature)
        node.add_feature(feature, next(iter(states)) if len(states) == 1 else list(states))
        num_scenarios *= len(states)
        unresolved_nodes += 1 if len(states) > 1 else 0
        node.del_feature(ps_feature)
    return num_scenarios, unresolved_nodes


def get_num_parsimonious_steps(tree, feature):
    ps_feature = get_personalized_feature_name(feature, PARS_STATES)
    ps_feature_num = get_personalized_feature_name(feature, PARS_STATE2NUM)

    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.add_feature(ps_feature_num, {state: 0 for state in getattr(node, ps_feature)})
        else:
            state2num = {}
            for state in getattr(node, ps_feature):
                num = 0
                for child in node.children:
                    child_state2num = getattr(child, ps_feature_num)
                    num += min(((0 if state == child_state else 1) + child_num)
                               for (child_state, child_num) in child_state2num.items())
                state2num[state] = num
            node.add_feature(ps_feature_num, state2num)
            for child in node.children:
                child.del_feature(ps_feature_num)
    state2num = getattr(tree, ps_feature_num)
    tree.del_feature(ps_feature_num)
    return min(state2num.values())
