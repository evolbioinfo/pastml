import logging

import pandas as pd
from ete3.parser.newick import write_newick

from pastml.tree import read_tree


def root_tree(tr, out_ids=None, in_ids=None, keep_outgroup=False):
    leaf_names = {l.name for l in tr.iter_leaves()}
    if out_ids:
        out_ids &= leaf_names
        in_ids = leaf_names - out_ids
    elif in_ids:
        in_ids &= leaf_names
        out_ids = leaf_names - in_ids
    else:
        raise ValueError('Either ingroup or outgroup ids must be specified!')
    n_out = len(out_ids)
    n_in = len(in_ids)
    logging.info('Trying to set {} elements as outgroup'.format(len(out_ids)))
    inverse = False
    if len(out_ids) == 1:
        outgroup_id = out_ids.pop()
        ancestor = next(l for l in tr.iter_leaves() if l.name == outgroup_id)
    else:
        ancestor = tr.get_common_ancestor(*out_ids)
        o_clu_len = len(ancestor.get_leaves())
        if o_clu_len != n_out:
            inverse = True
            ancestor = tr.get_common_ancestor(*in_ids)
            i_clu_len = len(ancestor.get_leaves())
            if i_clu_len != n_in:
                raise ValueError('The outgroup is incorporated inside the tree: '
                                 '%d outgroup sequences cluster inside a %d-leaf subtree, ' % (n_out, o_clu_len)
                                 + 'while %d sequences of interest cluster inside a %d-leaf subtree' % (
                                 n_in, i_clu_len))
    logging.info('%s:\n%s' % (('Our tree' if inverse else 'Ancestor'), ancestor.get_ascii()))
    tr.set_outgroup(ancestor)
    left, right = tr.children
    if ancestor not in tr.children:
        raise ValueError('The rerooting did not work out!!!')
    if keep_outgroup:
        logging.info('Keeping the outgroup')
        return tr
    if inverse:
        tr = ancestor
    else:
        tr = left if right == ancestor else right
    tr.dist = 0
    tr.up = None
    logging.info('The rooted tree contains %d leaves instead of %d' % (len(tr.get_leaves()), n_in + n_out))

    # If the root contains many children it will be considered as not rooted, so add a fake one if needed
    children = list(tr.children)
    if len(children) > 2:
        fake_child = tr.add_child(dist=0)
        for child in children[1:]:
            tr.remove_child(child)
            fake_child.add_child(child)
    return tr


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_tree', required=True, type=str)
    parser.add_argument('--output_tree', required=True, type=str)
    parser.add_argument('--ids', nargs='+', required=True, type=str)
    parser.add_argument('--ingroup', action='store_true')
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    tr = read_tree(params.input_tree)

    for group in params.ids:
        ids = set(pd.read_table(group, index_col=0, header=None).index.map(str))
        out_ids, in_ids = None, None
        if params.ingroup:
            in_ids = ids
        else:
            out_ids = ids
        tr = root_tree(tr, out_ids=out_ids, in_ids=in_ids)

    nwk = write_newick(tr, format_root_node=True, format=2)
    with open(params.output_tree, 'w+') as f:
        f.write('%s\n' % nwk)