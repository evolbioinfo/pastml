import logging
from collections import defaultdict

import numpy as np

from pastml.tree import DATE

REASONABLE_NUMBER_OF_TIPS = 15

CATEGORIES = 'categories'

NUM_TIPS_INSIDE = 'max_size'

TIPS_INSIDE = 'in_tips'
TIPS_BELOW = 'all_tips'

METACHILD = 'metachild'


def compress_tree(tree, columns, can_merge_diff_sizes=True, tip_size_threshold=REASONABLE_NUMBER_OF_TIPS):
    for n in tree.traverse('postorder'):
        n.add_feature(TIPS_INSIDE, defaultdict(list))
        n.add_feature(TIPS_BELOW, defaultdict(list))
        if n.is_leaf():
            getattr(n, TIPS_INSIDE)[getattr(n, DATE, 0)].append(n.name)
            getattr(n, TIPS_BELOW)[getattr(n, DATE, 0)].append(n.name)
        else:
            for _ in n:
                getattr(n, TIPS_BELOW)[getattr(_, DATE, 0)].append(_.name)

    collapse_vertically(tree, columns)
    remove_mediators(tree, columns)

    for n in tree.traverse():
        n.add_feature(NUM_TIPS_INSIDE, sum_len_values(getattr(n, TIPS_INSIDE)))
        n.add_feature(TIPS_INSIDE, [getattr(n, TIPS_INSIDE)])
        n.add_feature(TIPS_BELOW, [getattr(n, TIPS_BELOW)])

    get_bin = lambda _: _
    collapse_horizontally(tree, columns, get_bin)

    if can_merge_diff_sizes and len(tree) > tip_size_threshold:
        get_bin = lambda _: int(np.log10(max(1, _)))
        logging.getLogger('pastml').debug('Allowed merging nodes of different sizes.')
        collapse_horizontally(tree, columns, get_bin)

    if len(tree) > tip_size_threshold:
        for n in tree.traverse():
            n.add_feature(NUM_TIPS_INSIDE, sum(sum_len_values(_) for _ in getattr(n, TIPS_INSIDE))
                          / len(getattr(n, TIPS_INSIDE)))

        def get_tsize(n):
            res = getattr(n, NUM_TIPS_INSIDE) * len(getattr(n, TIPS_INSIDE))
            while n.up:
                n = n.up
                res *= len(getattr(n, TIPS_INSIDE))
            return res

        tip_sizes = [get_tsize(_) for _ in tree]
        if len(tip_sizes) > tip_size_threshold:
            threshold = sorted(tip_sizes)[-tip_size_threshold]
            logging.getLogger('pastml').debug('Set tip size threshold to {}.'.format(threshold))
            remove_small_tips(tree, to_be_removed=lambda _: get_tsize(_) < threshold)
            remove_mediators(tree, columns)

            collapse_horizontally(tree, columns, get_bin)
    return tree


def collapse_horizontally(tree, columns, tips2bin):
    config_cache = {}

    def get_configuration(n):
        if n not in config_cache:
            config_cache[n] = (len(getattr(n, TIPS_INSIDE)),
                               (tips2bin(getattr(n, NUM_TIPS_INSIDE)),
                                tuple(tuple(sorted(getattr(n, column, set()))) for column in columns),
                                tuple(sorted([get_configuration(_) for _ in n.children]))))
        return config_cache[n]

    collapsed_configurations = 0

    for n in tree.traverse('postorder'):
        config2children = defaultdict(list)
        for _ in n.children:
            config2children[get_configuration(_)[1]].append(_)
        for children in (_ for _ in config2children.values() if len(_) > 1):
            collapsed_configurations += 1
            children = sorted(children, key=lambda _: getattr(_, DATE, 0))
            child = children[0]
            for c in children[1:]:
                n.remove_child(c)
            child.add_feature(METACHILD, True)

            tips_inside, tips_below = [], []
            for _ in children:
                tips_inside.extend(getattr(_, TIPS_INSIDE))
                tips_below.extend(getattr(_, TIPS_BELOW))

            child.add_feature(TIPS_INSIDE, tips_inside)
            child.add_feature(TIPS_BELOW, tips_below)
            child.add_feature(NUM_TIPS_INSIDE, sum(sum_len_values(_) for _ in getattr(child, TIPS_INSIDE))
                              / len(getattr(child, TIPS_INSIDE)))
            child.add_feature(DATE, min(getattr(_, DATE, 0) for _ in children))
            if child in config_cache:
                config_cache[child] = (len(getattr(child, TIPS_INSIDE)), config_cache[child][1])

    logging.getLogger('pastml').debug(
        'Collapsed {} sets of equivalent configurations horizontally.'.format(collapsed_configurations))


def remove_small_tips(tree, to_be_removed):
    num_removed = 0
    changed = True
    while changed:
        changed = False
        for l in tree.get_leaves():
            parent = l.up
            if parent and to_be_removed(l):
                num_removed += 1
                parent.remove_child(l)
                parent.add_feature(DATE,
                                   min(min(_.keys()) if _ else 200000 for _ in getattr(parent, TIPS_INSIDE))
                                   if getattr(parent, TIPS_INSIDE) else 200000)
                if parent.children:
                    parent.add_feature(DATE, min(getattr(parent, DATE), min(getattr(_, DATE) for _ in parent.children)))
                changed = True
    logging.getLogger('pastml').debug(
        'Recursively removed {} tips of size smaller or equal to the threshold.'.format(num_removed))


def collapse_vertically(tree, columns):
    """
    Collapses a child node into its parent if they are in the same state.
    :param columns: a list of characters
    :param tree: ete3.Tree
    :return: void, modifies the input tree
    """

    def _same_states(node1, node2, columns):
        for column in columns:
            if getattr(node1, column, set()) != getattr(node2, column, set()):
                return False
        return True

    num_collapsed = 0
    for n in tree.traverse('postorder'):
        if n.is_leaf():
            continue

        children = list(n.children)
        for child in children:
            # merge the child into this node if their states are the same
            if _same_states(n, child, columns):
                for date, tip_names in getattr(child, TIPS_INSIDE).items():
                    getattr(n, TIPS_INSIDE)[date].extend(tip_names)

                n.remove_child(child)
                grandchildren = list(child.children)
                for grandchild in grandchildren:
                    n.add_child(grandchild)
                num_collapsed += 1
    logging.getLogger('pastml').debug('Collapsed vertically {} internal nodes where there was no state change.'
                                      .format(num_collapsed))


def remove_mediators(tree, columns):
    """
    Removes intermediate nodes that are just mediators between their parent and child states.
    :param columns: list of characters
    :param tree: ete3.Tree
    :return: void, modifies the input tree
    """
    num_removed = 0
    for n in tree.traverse('postorder'):
        if getattr(n, METACHILD, False) or n.is_leaf() or len(n.children) > 1 or not n.up:
            continue
        parent = n.up
        child = n.children[0]

        compatible = True
        for column in columns:
            states = getattr(n, column, set())
            parent_states = getattr(parent, column, set())
            child_states = getattr(child, column, set())
            # if mediator has unresolved states, it should hesitate between the parent and the child:
            if states != child_states | parent_states:
                compatible = False
                break

        if compatible:
            n_tips_inside = getattr(n, TIPS_INSIDE)[0] if isinstance(getattr(n, TIPS_INSIDE), list) \
                else getattr(n, TIPS_INSIDE)
            tips_inside = getattr(parent, TIPS_INSIDE)[0] if isinstance(getattr(parent, TIPS_INSIDE), list) \
                else getattr(parent, TIPS_INSIDE)
            for date, tip_names in n_tips_inside.items():
                tips_inside[date].extend(tip_names)

            parent.remove_child(n)
            parent.add_child(child)
            num_removed += 1
    if num_removed:
        logging.getLogger('pastml').debug("Removed {} internal nodes"
                                          " with the state unresolved between their parent's and their only child's."
                                          .format(num_removed))


def merge_features(main_node, nodes, features, op, default_value=0):
    for feature in features:
        main_node.add_feature(feature, op([getattr(node, feature, default_value) for node in nodes]))


def sum_len_values(kv_dict):
    return sum(len(_) for _ in kv_dict.values())