import logging
from ete3.parser.newick import write_newick

from pastml.tree import read_tree

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_tree', required=True, type=str)
    parser.add_argument('--output_tree', required=True, type=str)
    parser.add_argument('--threshold', required=True, type=str)
    parser.add_argument('--feature', required=True, type=str)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    tr = read_tree(params.input_tree)

    try:
        threshold = float(params.threshold)
    except:
        # may be it's a string threshold then
        threshold = params.threshold

    num_collapsed = 0
    for n in list(tr.traverse('postorder')):
        if not n.is_root():
            for child in n.children:
                if not child.is_leaf() and getattr(child, params.feature) <= threshold:
                    n.remove_child(child)
                    for grandchild in child.children:
                        n.add_child(grandchild)
                    num_collapsed += 1
    logging.info('Collapsed {} branches with {} <= {}'.format(num_collapsed, params.feature, threshold))
    nwk = write_newick(tr, format_root_node=True, format=1)
    with open(params.output_tree, 'w+') as f:
        f.write('%s\n' % nwk)
