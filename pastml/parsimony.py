import logging
from collections import Counter

from pastml import get_personalized_feature_name, METHOD, STATES, CHARACTER, NUM_SCENARIOS, NUM_UNRESOLVED_NODES, \
    NUM_NODES, NUM_TIPS, NUM_STATES_PER_NODE, PERC_UNRESOLVED

STEPS = 'steps'

DOWNPASS = 'DOWNPASS'
ACCTRAN = 'ACCTRAN'
DELTRAN = 'DELTRAN'
MP = 'MP'

MP_METHODS = {DOWNPASS, ACCTRAN, DELTRAN}
META_MP_METHODS = {MP}

BU_PARS_STATES = 'BOTTOM_UP_PARSIMONY'
TD_PARS_STATES = 'TOP_DOWN_PARSIMONY'
PARS_STATES = 'PARSIMONY'
PARS_STATE2NUM = 'PARSIMONY_STEPS'


def is_meta_mp(method):
    """
    Checks if the method is a meta max parsimony method, combining several methods, i.e. MP.

    :param method: prediction method
    :type method: str
    :return: bool
    """
    return method in META_MP_METHODS


def get_default_mp_method():
    return DOWNPASS


def is_parsimonious(method):
    """
    Checks if the method is max likelihood, i.e. is either joint or one of the marginal ones
    (marginal itself, or MAP, or MPPA).
    :param method: str, the ancestral state prediction method used by PASTML.
    :return: bool
    """
    return method in MP_METHODS | {MP}


def initialise_parsimonious_states(tree, feature, states):
    """
    Initializes the bottom-up state arrays for tips based on their states given by the feature.

    :param tree: ete3.Tree, tree for which the tip states are to be initialized
    :param feature: str, feature in which the tip states are stored (the value could be None for a missing state)
    :param states: numpy array, possible states.
    :return: void, adds the get_personalised_feature_name(feature, BU_PARS) feature to tree tips.
    """
    ps_feature_down = get_personalized_feature_name(feature, BU_PARS_STATES)
    ps_feature = get_personalized_feature_name(feature, PARS_STATES)
    all_states = set(states)

    for node in tree.traverse():
        state = getattr(node, feature, set())
        if not state:
            node.add_feature(ps_feature_down, all_states)
        else:
            node.add_feature(ps_feature_down, state)
        node.add_feature(ps_feature, getattr(node, ps_feature_down))


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
            children_states = get_most_common_states(getattr(child, ps_feature) for child in node.children)
            node_states = getattr(node, ps_feature)
            state_intersection = node_states & children_states
            node.add_feature(ps_feature, state_intersection if state_intersection else node_states)


def acctran(tree, character, feature=PARS_STATES):
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
    :param character: str, character for which the parsimonious states are reconstructed
    :return: void, adds get_personalized_feature_name(feature, PARS_STATES) feature to the tree nodes
    """

    ps_feature_down = get_personalized_feature_name(character, BU_PARS_STATES)

    for node in tree.traverse('preorder'):
        if node.is_root():
            node.add_feature(feature, getattr(node, ps_feature_down))
        node_states = getattr(node, feature)
        for child in node.children:
            child_states = getattr(child, ps_feature_down)
            state_intersection = node_states & child_states
            child.add_feature(feature, state_intersection if state_intersection else child_states)


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
        down_up_states = get_most_common_states([getattr(node, ps_feature_up)]
                                                + [getattr(child, ps_feature_down) for child in node.children]) \
            if not node.is_leaf() else getattr(node, ps_feature_up)
        preset_states = getattr(node, ps_feature)
        state_intersection = down_up_states & preset_states
        node.add_feature(ps_feature, state_intersection if state_intersection else preset_states)

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


def parsimonious_acr(tree, character, prediction_method, states, num_nodes, num_tips):
    """
    Calculates parsimonious states on the tree and stores them in the corresponding feature.

    :param states: numpy array of possible states
    :param prediction_method: str, ACCTRAN (accelerated transformation), DELTRAN (delayed transformation) or DOWNPASS
    :param tree: ete3.Tree, the tree of interest
    :param character: str, character for which the parsimonious states are reconstructed
    :return: dict, mapping between reconstruction parameters and values
    """
    initialise_parsimonious_states(tree, character, states)
    uppass(tree, character)

    results = []
    result = {STATES: states, NUM_NODES: num_nodes, NUM_TIPS: num_tips}

    logger = logging.getLogger('pastml')

    def process_result(method, feature):
        out_feature = get_personalized_feature_name(character, method) if prediction_method != method else character
        res = result.copy()
        res[NUM_SCENARIOS], res[NUM_UNRESOLVED_NODES], res[NUM_STATES_PER_NODE] \
            = choose_parsimonious_states(tree, feature, out_feature)
        res[NUM_STATES_PER_NODE] /= num_nodes
        res[PERC_UNRESOLVED] = res[NUM_UNRESOLVED_NODES] * 100 / num_nodes
        logger.debug('{} node{} unresolved ({:.2f}%) for {} by {}, '
                     'i.e. {:.4f} state{} per node in average.'
                     .format(res[NUM_UNRESOLVED_NODES], 's are' if res[NUM_UNRESOLVED_NODES] != 1 else ' is',
                             res[PERC_UNRESOLVED], character, method,
                             res[NUM_STATES_PER_NODE], 's' if res[NUM_STATES_PER_NODE] > 1 else ''))
        res[CHARACTER] = out_feature
        res[METHOD] = method
        results.append(res)

    if prediction_method in {ACCTRAN, MP}:
        feature = get_personalized_feature_name(character, PARS_STATES)
        if prediction_method == MP:
            feature = get_personalized_feature_name(feature, ACCTRAN)
        acctran(tree, character, feature)
        result[STEPS] = get_num_parsimonious_steps(tree, feature)
        process_result(ACCTRAN, feature)

        bu_feature = get_personalized_feature_name(character, BU_PARS_STATES)
        for node in tree.traverse():
            if prediction_method == ACCTRAN:
                node.del_feature(bu_feature)
            node.del_feature(feature)

    if prediction_method != ACCTRAN:
        downpass(tree, character, states)
        feature = get_personalized_feature_name(character, PARS_STATES)
        if prediction_method == DOWNPASS:
            result[STEPS] = get_num_parsimonious_steps(tree, feature)
        if prediction_method in {DOWNPASS, MP}:
            process_result(DOWNPASS, feature)
        if prediction_method in {DELTRAN, MP}:
            deltran(tree, character)
            if prediction_method == DELTRAN:
                result[STEPS] = get_num_parsimonious_steps(tree, feature)
            process_result(DELTRAN, feature)
        for node in tree.traverse():
            node.del_feature(feature)

    logger.debug("Parsimonious reconstruction for {} requires {} state changes."
                 .format(character, result[STEPS]))
    return results


def choose_parsimonious_states(tree, ps_feature, out_feature):
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
    num_states = 0
    for node in tree.traverse():
        states = getattr(node, ps_feature)
        node.add_feature(out_feature, states)
        n = len(states)
        num_scenarios *= n
        unresolved_nodes += 1 if n > 1 else 0
        num_states += n
    return num_scenarios, unresolved_nodes, num_states


def get_num_parsimonious_steps(tree, feature):
    ps_feature_num = get_personalized_feature_name(feature, PARS_STATE2NUM)

    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.add_feature(ps_feature_num, {state: 0 for state in getattr(node, feature)})
        else:
            state2num = {}
            for state in getattr(node, feature):
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
