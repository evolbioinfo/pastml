from ete3 import Tree

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    params = parser.parse_args()

    tree = Tree(params.input, format=3)
    tree.resolve_polytomy()
    tree.write(outfile=params.output, format=5)
