import os

from pastml.all_in_one import pastml_pipeline
from pastml.acr.maxlikelihood.ml import MPPA

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_phyml_tree.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata.tab')


if '__main__' == __name__:
    mutations = ['RT:M184V']
    pastml_pipeline(data=STATES_INPUT,
                    tree=TREE_NWK, prediction_method=MPPA,
                    html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}.html'.format('_'.join(mutations))),
                    verbose=True, columns=mutations, upload_to_itol=True)
