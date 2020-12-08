import os

from pastml.acr import pastml_pipeline
from pastml.ml import MPPA

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_phyml_tree.nwk')
TREE_NEXUS = os.path.join(DATA_DIR, 'best', 'phyml.lsd2.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata.tab')
WD = os.path.join(DATA_DIR, 'pastml_params', '{}')


if '__main__' == __name__:
    mutations = ['RT:M184V']

    pastml_pipeline(data=STATES_INPUT, prediction_method=MPPA,
                    tree=TREE_NEXUS, html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}_skyline.html'.format('_'.join(mutations))),
                    verbose=True, columns=mutations, skyline=[1991], tip_size_threshold=100,
                    work_dir=WD.format('skyline_1991'))
    pastml_pipeline(data=STATES_INPUT, prediction_method=MPPA,
                    tree=TREE_NEXUS, html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}_noskyline.html'.format('_'.join(mutations))),
                    verbose=True, columns=mutations, tip_size_threshold=100,
                    work_dir=WD.format('no_skyline'))
