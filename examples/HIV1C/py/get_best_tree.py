from shutil import copyfile


def get_best_tree(trees, logs, type):

    def get_lh(line):
        if 'fast' == type:
            if 'Max-lk operations:' in line:
                return float(line[line.find('posterior') + len('posterior'): line.find('star-only')])
            return None
        if 'phyml' == type:
            if 'Log-likelihood:' in line:
                return float(line[line.find('Log-likelihood:') + len('Log-likelihood:'):])
            return None
        if 'raxml' == type:
            if 'Final LogLikelihood:' in line:
                return float(line[line.find('Final LogLikelihood:') + len('Final LogLikelihood:'):])
            return None

    best_tree, best_log = None, None
    best_lh = None
    for tree, log in zip(trees, logs):
        with open(log, 'r') as f:
            for line in f:
                lh = get_lh(line)
                if lh is not None:
                    if best_lh is None or lh > best_lh:
                        best_lh = lh
                        best_tree, best_log = tree, log
                    break
    return best_tree, best_log


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--trees', required=True, type=str, nargs='+')
    parser.add_argument('--logs', required=True, type=str, nargs='+')
    parser.add_argument('--out_tree', required=True, type=str)
    parser.add_argument('--out_log', required=True, type=str)
    parser.add_argument('--type', required=True, type=str, choices=('phyml', 'raxml', 'fast'))
    params = parser.parse_args()

    tree, log = get_best_tree(params.trees, params.logs, params.type)

    copyfile(tree, params.out_tree)
    copyfile(log, params.out_log)
