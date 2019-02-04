import logging
import os

import pandas as pd
from ete3 import Tree

from pastml import get_personalized_feature_name
from pastml.acr import pastml_pipeline
from pastml.file import get_combined_ancestral_state_file, get_pastml_parameter_file
from pastml.ml import MPPA, F81, ML
from pastml.tree import remove_certain_leaves

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'best', 'phyml_tree.rooted.collapsed_dist_0.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata_loc.tab')


if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',)

    mutation = 'RT:M184V'
    pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                    html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}.html'.format(mutation)),
                    verbose=True, columns=[mutation],
                    work_dir=os.path.join(DATA_DIR, 'pastml', 'DRM_{}'.format(mutation)))