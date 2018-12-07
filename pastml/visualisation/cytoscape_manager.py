import logging
import os
from queue import Queue

import numpy as np
from jinja2 import Environment, PackageLoader

from pastml.visualisation.colour_generator import get_enough_colours, WHITE
from pastml.visualisation.tree_compressor import NUM_TIPS_INSIDE, sum_len_values, TIPS_INSIDE, TIPS_BELOW, \
    REASONABLE_NUMBER_OF_TIPS, compress_tree
from pastml.tree import DATE, DEPTH, LEVEL

TIP_LIMIT = 1000

MIN_EDGE_SIZE = 50
MIN_FONT_SIZE = 80
MIN_NODE_SIZE = 200

UNRESOLVED = 'unresolved'
TIP = 'tip'

TOOLTIP = 'tooltip'

DATA = 'data'
ID = 'id'
EDGES = 'edges'
NODES = 'nodes'
ELEMENTS = 'elements'

NODE_SIZE = 'node_size'
NODE_NAME = 'node_name'
EDGE_SIZE = 'edge_size'
EDGE_NAME = 'edge_name'
FONT_SIZE = 'node_fontsize'


def get_fake_node(n_id):
    attributes = {ID: n_id, 'fake': 1}
    return _get_node(attributes)


def get_node(n, n_id, tooltip='', clazz=None):
    features = {feature: getattr(n, feature) for feature in n.features if feature in [DATE, UNRESOLVED]
                or feature.startswith('node_')}
    features[ID] = n_id
    if n.is_leaf():
        features[TIP] = 1
    features[TOOLTIP] = tooltip
    return _get_node(features, clazz=_clazz_list2css_class(clazz))


def get_edge(source_name, target_name, **kwargs):
    return _get_edge(source=source_name, target=target_name, **kwargs)


def get_scaling_function(y_m, y_M, x_m, x_M):
    """
    Returns a linear function y = k x + b, where y \in [m, M]
    :param y_m:
    :param y_M:
    :param x_m:
    :param x_M:
    :return:
    """
    if x_M <= x_m:
        return lambda _: y_m
    k = (y_M - y_m) / (x_M - x_m)
    b = y_m - k * x_m
    return lambda _: int(k * _ + b)


def set_cyto_features_compressed(n, tips_inside, tips_below, size_scaling, e_size_scaling, font_scaling, transform_size,
                                 transform_e_size, state, suffix=''):
    min_n_tips = min(sum_len_values(_) for _ in tips_inside) if tips_inside else 0
    max_n_tips = max(sum_len_values(_) for _ in tips_inside) if tips_inside else 0

    min_n_tips_below = min(sum_len_values(_) for _ in tips_below) if tips_below else 0
    max_n_tips_below = max(sum_len_values(_) for _ in tips_below) if tips_below else 0

    tips_inside_str = ' {}'.format('{}-{}'.format(min_n_tips, max_n_tips) if min_n_tips != max_n_tips else min_n_tips) \
        if max_n_tips > 0 else ' 0'
    tips_below_str = ' {}'.format('{}-{}'.format(min_n_tips_below, max_n_tips_below)
                                  if min_n_tips_below != max_n_tips_below else min_n_tips_below) \
        if max_n_tips_below > 0 else ' 0'

    n.add_feature('{}{}'.format(NODE_NAME, suffix), '{}{}'.format(state, tips_inside_str))
    size_factor = 2 if getattr(n, UNRESOLVED, False) else 1
    n.add_feature('{}{}'.format(NODE_SIZE, suffix),
                  (size_scaling(transform_size(max_n_tips)) if max_n_tips else int(MIN_NODE_SIZE / 1.5)) * size_factor)
    n.add_feature('{}{}'.format(FONT_SIZE, suffix),
                  font_scaling(transform_size(max_n_tips)) if max_n_tips else MIN_FONT_SIZE)

    if max_n_tips > 0 or max_n_tips_below > 0:
        n.add_feature('node_{}{}'.format(TIPS_INSIDE, suffix), tips_inside_str)
        n.add_feature('node_{}{}'.format(TIPS_BELOW, suffix), tips_below_str)

        edge_size = max(len(tips_inside), 1)
        if edge_size > 1:
            n.add_feature('edge_meta{}'.format(suffix), 1)
            n.add_feature('node_meta{}'.format(suffix), 1)
        n.add_feature('{}{}'.format(EDGE_NAME, suffix), str(edge_size) if edge_size != 1 else '')
        e_size = e_size_scaling(transform_e_size(edge_size))
        n.add_feature('{}{}'.format(EDGE_SIZE, suffix), e_size)


def set_cyto_features_tree(n, state):
    n.add_feature(NODE_NAME, state)
    n.add_feature(EDGE_NAME, '{:.3f}'.format(n.dist))


def _tree2json(tree, column2states, name_feature, node2tooltip, years=None, is_compressed=True):
    e_size_scaling, font_scaling, size_scaling, transform_e_size, transform_size = get_size_transformations(tree)

    for n in tree.traverse():
        state = get_column_value_str(n, name_feature, format_list=False, list_value='') if name_feature else ''
        n.add_feature('node_root_id', n.name)

        if is_compressed:
            tips_inside, tips_below = getattr(n, TIPS_INSIDE, []), getattr(n, TIPS_BELOW, [])
            if isinstance(tips_inside, dict):
                tips_inside = [tips_inside]
            if isinstance(tips_below, dict):
                tips_below = [tips_below]

            set_cyto_features_compressed(n, tips_inside, tips_below, size_scaling, e_size_scaling, font_scaling,
                                         transform_size, transform_e_size, state)

            if years and len(years) > 1:
                i = len(years) - 1
                while i >= 0:
                    suffix = '_{}'.format(i)
                    tips_inside = [e for e in
                                   ({k: _ for (k, _) in year2tips.items() if k <= i} for year2tips in tips_inside)
                                   if e]
                    tips_below = [e for e in
                                  ({k: _ for (k, _) in year2tips.items() if k <= i} for year2tips in tips_below)
                                  if e]
                    set_cyto_features_compressed(n, tips_inside, tips_below, size_scaling, e_size_scaling, font_scaling,
                                                 transform_size, transform_e_size, state, suffix=suffix)
                    i -= 1
        else:
            set_cyto_features_tree(n, state)

    clazzes = set()
    nodes, edges = [], []

    num_tips = len(tree)
    if not is_compressed and num_tips <= TIP_LIMIT:
        dist_step = np.median([getattr(n, DEPTH) / getattr(n, LEVEL) for n in tree.traverse() if not n.is_root()]) \
                    / (3 if num_tips <= TIP_LIMIT / 2 else 2)

    todo = Queue()
    todo.put_nowait(tree)
    node2id = {tree: 0}
    i = 1

    n2sort_name = {}
    for node in tree.traverse('preorder'):
        if node.is_root():
            n2sort_name[node] = (node.name,)
        else:
            n2sort_name[node] = (*n2sort_name[node.up], node.name)

    sort_key = lambda n: n2sort_name[n]
    while not todo.empty():
        n = todo.get_nowait()
        for c in sorted(n.children, key=sort_key):
            node2id[c] = i
            i += 1
            todo.put_nowait(c)

    one_column = next(iter(column2states.keys())) if len(column2states) == 1 else None

    for n, n_id in sorted(node2id.items(), key=lambda ni: ni[1]):
        if n == tree and not is_compressed and num_tips <= TIP_LIMIT and int(n.dist / dist_step) > 0:
            fake_id = 'fake_node_{}'.format(n_id)
            nodes.append(get_fake_node(fake_id))
            edges.append(get_edge(fake_id, n_id, minLen=int(n.dist / dist_step),
                                  **{feature: getattr(n, feature) for feature in n.features
                                     if feature.startswith('edge_') or feature == DATE}))
        if one_column:
            values = getattr(n, one_column, set())
            clazz = tuple(sorted(values))
        else:
            clazz = tuple('{}_{}'.format(column, get_column_value_str(n, column, format_list=False, list_value=''))
                          for column in sorted(column2states.keys()))
        if clazz:
            clazzes.add(clazz)
        nodes.append(get_node(n, n_id, tooltip=node2tooltip[n], clazz=clazz))

        for child in sorted(n.children, key=lambda _: node2id[_]):
            edge_attributes = {feature: getattr(child, feature) for feature in child.features
                               if feature.startswith('edge_') or feature == DATE}
            source_name = n_id
            if not is_compressed and num_tips <= TIP_LIMIT:
                target_name = 'fake_node_{}'.format(node2id[child])
                nodes.append(get_fake_node(target_name))
                edges.append(get_edge(source_name, target_name, fake=1,
                                      **{k: v for (k, v) in edge_attributes.items() if EDGE_NAME not in k}))
                source_name = target_name
                if int(n.dist / dist_step) > 0:
                    edge_attributes['minLen'] = int(child.dist / dist_step)
            edges.append(get_edge(source_name, node2id[child], **edge_attributes))

    json_dict = {NODES: nodes, EDGES: edges}
    return json_dict, sorted(clazzes)


def get_size_transformations(tree):
    n_sizes = [getattr(n, NUM_TIPS_INSIDE) for n in tree.traverse() if getattr(n, NUM_TIPS_INSIDE, False)]
    max_size = max(n_sizes) if n_sizes else 1
    min_size = min(n_sizes) if n_sizes else 1
    need_log = max_size / min_size > 100
    transform_size = lambda _: np.power(np.log10(_ + 9) if need_log else _, 1 / 2)

    e_szs = [len(getattr(n, TIPS_INSIDE)) for n in tree.traverse() if getattr(n, TIPS_INSIDE, False)]
    max_e_size = max(e_szs) if e_szs else 1
    min_e_size = min(e_szs) if e_szs else 1
    need_e_log = max_e_size / min_e_size > 100
    transform_e_size = lambda _: np.log10(_) if need_e_log else _

    size_scaling = get_scaling_function(y_m=MIN_NODE_SIZE, y_M=MIN_NODE_SIZE * min(8, int(max_size / min_size)),
                                        x_m=transform_size(min_size), x_M=transform_size(max_size))
    font_scaling = get_scaling_function(y_m=MIN_FONT_SIZE, y_M=MIN_FONT_SIZE * min(3, int(max_size / min_size)),
                                        x_m=transform_size(min_size), x_M=transform_size(max_size))
    e_size_scaling = get_scaling_function(y_m=MIN_EDGE_SIZE, y_M=MIN_EDGE_SIZE * min(3, int(max_e_size / min_e_size)),
                                          x_m=transform_e_size(min_e_size), x_M=transform_e_size(max_e_size))

    return e_size_scaling, font_scaling, size_scaling, transform_e_size, transform_size


def save_as_cytoscape_html(tree, out_html, column2states, layout='dagre', name_feature='name',
                           name2colour=None, n2tooltip=None, years=None, is_compressed=True):
    """
    Converts a tree to an html representation using Cytoscape.js.

    If categories are specified they are visualised as pie-charts inside the nodes,
    given that each node contains features corresponding to these categories with values being the percentage.
    For instance, given categories ['A', 'B', 'C'], a node with features {'A': 50, 'B': 50}
    will have a half-half pie-chart (half-colored in a colour of A, and half B).

    If dist_step is specified, the edges are rescaled accordingly to their dist (node.dist / dist_step),
    otherwise all edges are drawn of the same length.

    otherwise all edges are drawn of the same length.
    :param name_feature: str, a node feature whose value will be used as a label
    returns a key to be used for sorting nodes on the same level in the tree.
    :param n2tooltip: dict, TreeNode to str mapping tree nodes to tooltips.
    :param layout: str, name of the layout for Cytoscape.js
    :param name2colour: dict, str to str, category name to HEX colour mapping 
    :param categories: a list of categories for the pie-charts inside the nodes
    :param tree: ete3.Tree
    :param out_html: path where to save the resulting html file.
    """
    graph_name = os.path.splitext(os.path.basename(out_html))[0]

    json_dict, clazzes \
        = _tree2json(tree, column2states, name_feature=name_feature,
                     node2tooltip=n2tooltip, years=years, is_compressed=is_compressed)
    env = Environment(loader=PackageLoader('pastml'))
    template = env.get_template('pie_tree.js') if is_compressed else env.get_template('pie_tree_simple.js')

    clazz2css = {}
    for clazz_list in clazzes:
        n = len(clazz_list)
        css = ''
        for i, cat in enumerate(clazz_list, start=1):
            css += """
                'pie-{i}-background-color': "{colour}",
                'pie-{i}-background-size': '{percent}\%',
            """.format(i=i, percent=round(100 / n, 2), colour=name2colour[cat])
        clazz2css[_clazz_list2css_class(clazz_list)] = css
    graph = template.render(clazz2css=clazz2css.items(), elements=json_dict, layout=layout, title=graph_name,
                            years=['{:g}'.format(_) for _ in years])
    slider = env.get_template('time_slider.html').render(min_date=getattr(tree, DATE), max_date=len(years) - 1) \
        if len(years) > 1 else ''

    template = env.get_template('index.html')
    page = template.render(graph=graph, title=graph_name, slider=slider)

    os.makedirs(os.path.abspath(os.path.dirname(out_html)), exist_ok=True)
    with open(out_html, 'w+') as fp:
        fp.write(page)


def _clazz_list2css_class(clazz_list):
    if not clazz_list:
        return None
    return ''.join(c for c in '-'.join(clazz_list) if c.isalnum() or '-' == c)


def _get_node(data, position=None, clazz=None):
    res = {DATA: data}
    if clazz:
        res['classes'] = clazz
    if position:
        res['position'] = position
    return res


def _get_edge(**data):
    return {DATA: data}


def get_column_value_str(n, column, format_list=True, list_value='<unresolved>'):
    values = getattr(n, column, set())
    if isinstance(values, str):
        return values
    return ' or '.join(sorted(values)) if format_list or len(values) == 1 else list_value


def visualize(tree, column2states, name_column=None, html=None, html_compressed=None,
              tip_size_threshold=REASONABLE_NUMBER_OF_TIPS, years=None):
    one_column = next(iter(column2states.keys())) if len(column2states) == 1 else None

    name2colour = {}
    for column, states in column2states.items():
        num_unique_values = len(states)
        colours = get_enough_colours(num_unique_values)
        for value, col in zip(states, colours):
            name2colour[value if one_column else '{}_{}'.format(column, value)] = col
        logging.getLogger('pastml').debug('Mapped states to colours for {} as following: {} -> {}.'
                                          .format(column, states, colours))
        # let ambiguous values be white
        if one_column is None:
            name2colour['{}_'.format(column)] = WHITE
        if column == name_column:
            state2color = dict(zip(states, colours))
            for n in tree.traverse():
                sts = getattr(n, column, set())
                if len(sts) == 1 and not n.is_root() and getattr(n.up, column, set()) == sts:
                    n.add_feature('edge_color', state2color[next(iter(sts))])

    years = sorted(years)

    def binary_search(start, end, value, array):
        if start >= end - 1:
            return start
        i = int((start + end) / 2)
        if array[i] == value or array[i] > value and (i == start or value > array[i - 1]):
            return i
        if array[i] > value:
            return binary_search(start, i, value, array)
        return binary_search(i + 1, end, value, array)

    # set internal node dates to min of its tips' dates
    for node in tree.traverse('postorder'):
        if not years or len(years) == 1:
            node.add_feature(DATE, 0)
        else:
            if node.is_leaf():
                date = getattr(node, DATE, None)
                if not date:
                    slider = len(years) - 1
                else:
                    slider = binary_search(0, len(years), date, years)
                node.add_feature(DATE, slider)
            else:
                node.add_feature(DATE, min(getattr(_, DATE) for _ in node.children))

        for column in column2states.keys():
            if len(getattr(node, column, set())) != 1:
                node.add_feature(UNRESOLVED, 1)

    def get_category_str(n):
        return '<br>'.join('{}: {}'.format(column, get_column_value_str(n, column, format_list=True))
                           for column in sorted(column2states.keys()))

    if html:
        save_as_cytoscape_html(tree, html, column2states=column2states,
                               name2colour=name2colour,
                               n2tooltip={n: get_category_str(n) for n in tree.traverse()},
                               name_feature='name', years=years, is_compressed=False)

    if html_compressed:
        tree = compress_tree(tree, columns=column2states.keys(), tip_size_threshold=tip_size_threshold)
        save_as_cytoscape_html(tree, html_compressed, column2states=column2states,
                               name2colour=name2colour, n2tooltip={n: get_category_str(n) for n in tree.traverse()},
                               years=years, name_feature=name_column, is_compressed=True)
    return tree
