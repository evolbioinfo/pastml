import os

from pastml.acr import pastml_pipeline
from pastml.ml import ML, MAP
from pastml.tree import read_forest

import pandas as pd

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_phyml_tree.nwk')
TREE_NEXUS = os.path.join(DATA_DIR, 'best', 'phyml.lsd2.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata.tab')


if '__main__' == __name__:
    mutations = ['RT:M184V']

    # df = ps.read_csv()
    # tree = read_forest(TREE_NEXUS)[0]
    # for n in tree.traverse():
    #     if
    pastml_pipeline(data=STATES_INPUT, prediction_method=MAP,
                    tree=TREE_NEXUS, html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}.html'.format('_'.join(mutations))),
                    html=os.path.join(DATA_DIR, 'maps', 'tree_{}.html'.format('_'.join(mutations))),
                    verbose=True, columns=mutations, skyline=[1991], tip_size_threshold=100)
