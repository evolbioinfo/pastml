from ete3.parser.newick import write_newick

from pastml.tree import read_tree, name_tree

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_tree', required=True, type=str)
    parser.add_argument('--output_tree', required=True, type=str)
    params = parser.parse_args()

    tr = read_tree(params.input_tree)
    name_tree(tr)

    nwk = write_newick(tr, format_root_node=True, format=3)
    with open(params.output_tree, 'w+') as f:
        f.write('%s\n' % nwk)
