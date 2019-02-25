import logging
from collections import defaultdict

import numpy as np

REASONABLE_NUMBER_OF_TIPS = 15

CATEGORIES = 'categories'

NUM_TIPS_INSIDE = 'max_size'

TIPS_INSIDE = 'in_tips'
INTERNAL_NODES_INSIDE = 'in_ns'
TIPS_BELOW = 'all_tips'
NAMES = 'names'

METACHILD = 'metachild'


def compress_tree(tree, columns, n2date, can_merge_diff_sizes=True, tip_size_threshold=REASONABLE_NUMBER_OF_TIPS):
    for n in tree.traverse('postorder'):
        n.add_feature(TIPS_INSIDE, [])
        n.add_feature(INTERNAL_NODES_INSIDE, [])
        n.add_feature(TIPS_BELOW, [])
        n.add_feature(NAMES, [n.name])
        if n.is_leaf():
            getattr(n, TIPS_INSIDE).append(n.name)
            getattr(n, TIPS_BELOW).append(n.name)
        else:
            getattr(n, INTERNAL_NODES_INSIDE).append(n.name)
            for _ in n:
                getattr(n, TIPS_BELOW).append(_.name)

    collapse_vertically(tree, columns)

    for n in tree.traverse():
        n.add_feature(NUM_TIPS_INSIDE, len(getattr(n, TIPS_INSIDE)))
        n.add_feature(TIPS_INSIDE, [getattr(n, TIPS_INSIDE)])
        n.add_feature(INTERNAL_NODES_INSIDE, [getattr(n, INTERNAL_NODES_INSIDE)])
        n.add_feature(TIPS_BELOW, [getattr(n, TIPS_BELOW)])

    remove_mediators(tree, columns)

    get_bin = lambda _: _
    collapse_horizontally(tree, columns, get_bin, n2date)

    if can_merge_diff_sizes and len(tree) > tip_size_threshold:
        get_bin = lambda _: int(np.log10(max(1, _)))
        logging.getLogger('pastml').debug('Allowed merging nodes of different sizes.')
        collapse_horizontally(tree, columns, get_bin, n2date)

    if len(tree) > tip_size_threshold:
        for n in tree.traverse('preorder'):
            multiplier = (getattr(n.up, 'multiplier') if n.up else 1) * len(getattr(n, TIPS_BELOW))
            n.add_feature('multiplier', multiplier)

        def get_tsize(n):
            return getattr(n, NUM_TIPS_INSIDE) * getattr(n, 'multiplier')

        tip_sizes = []
        for n in tree.traverse('postorder'):
            children_bs = 0 if not n.children else max(get_tsize(_) for _ in n.children)
            bs = get_tsize(n)
            if not n.is_root() and bs > children_bs:
                tip_sizes.append(bs)
        if len(tip_sizes) > tip_size_threshold:
            threshold = sorted(tip_sizes)[-tip_size_threshold]
            if min(tip_sizes) >= threshold:
                logging.getLogger('pastml')\
                    .debug('No tip is smaller than the threshold ({}, the size of the {}-th largest tip).'
                           .format(threshold, tip_size_threshold))
            else:
                logging.getLogger('pastml').debug('Set tip size threshold to {} (the size of the {}-th largest tip).'
                                                  .format(threshold, tip_size_threshold))
                remove_small_tips(tree, to_be_removed=lambda _: get_tsize(_) < threshold, n2date=n2date)
                remove_mediators(tree, columns)
                collapse_horizontally(tree, columns, get_bin, n2date)
    return tree


def collapse_horizontally(tree, columns, tips2bin, n2date):
    config_cache = {}

    def get_configuration(n):
        if n not in config_cache:
            # Configuration is (branch_width, (states, child_configurations)),
            # where branch_width is only used for recursive calls and is ignored when considering a merge
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
            children = sorted(children, key=lambda _: n2date[_.name])
            child = children[0]
            for sibling in children[1:]:
                getattr(child, TIPS_INSIDE).extend(getattr(sibling, TIPS_INSIDE))
                getattr(child, INTERNAL_NODES_INSIDE).extend(getattr(sibling, INTERNAL_NODES_INSIDE))
                getattr(child, TIPS_BELOW).extend(getattr(sibling, TIPS_BELOW))
                getattr(child, NAMES).extend(getattr(sibling, NAMES))
                n.remove_child(sibling)
            child.add_feature(METACHILD, True)
            child.add_feature(NUM_TIPS_INSIDE,
                              sum(len(_) for _ in getattr(child, TIPS_INSIDE)) / len(getattr(child, TIPS_INSIDE)))
            if child in config_cache:
                config_cache[child] = (len(getattr(child, TIPS_INSIDE)), config_cache[child][1])
    if collapsed_configurations:
        logging.getLogger('pastml').debug(
            'Collapsed {} sets of equivalent configurations horizontally.'.format(collapsed_configurations))


def remove_small_tips(tree, to_be_removed, n2date):
    num_removed = 0
    changed = True
    while changed:
        changed = False
        for l in tree.get_leaves():
            parent = l.up
            if parent and to_be_removed(l):
                num_removed += 1
                parent.remove_child(l)
                changed = True
    for n in tree.traverse('postorder'):
        date = min(min(n2date[_] for _ in tips) for tips in getattr(n, TIPS_INSIDE) if tips) \
            if getattr(n, NUM_TIPS_INSIDE) else None
        for c in n.children:
            date = n2date[c.name] if date is None else min(date, n2date[c.name])
        n2date[n.name] = date
    logging.getLogger('pastml').debug(
        'Recursively removed {} tips of size smaller than the threshold.'.format(num_removed))


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
                getattr(n, TIPS_INSIDE).extend(getattr(child, TIPS_INSIDE))
                getattr(n, INTERNAL_NODES_INSIDE).extend(getattr(child, INTERNAL_NODES_INSIDE))

                n.remove_child(child)
                grandchildren = list(child.children)
                for grandchild in grandchildren:
                    n.add_child(grandchild)
                num_collapsed += 1
    if num_collapsed:
        logging.getLogger('pastml').debug('Collapsed vertically {} internal nodes without state change.'
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
        if getattr(n, METACHILD, False) or n.is_leaf() or len(n.children) > 1 or n.is_root() \
                or getattr(n, NUM_TIPS_INSIDE) > 0:
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
            parent.remove_child(n)
            parent.add_child(child)
            num_removed += 1
    if num_removed:
        logging.getLogger('pastml').debug("Removed {} internal node{}"
                                          " with the state unresolved between the parent's and the only child's."
                                          .format(num_removed, '' if num_removed == 1 else 's'))
