import os

from pastml.all_in_one import pastml_pipeline
from pastml.acr.maxlikelihood.ml import MPPA

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'real', 'raxml_tree.dated.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'real', 'metadata.tab')

if '__main__' == __name__:
    character = 'Location'
    pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK, prediction_method=MPPA,
                    html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}.html'.format(character)),
                    html=os.path.join(DATA_DIR, 'maps', 'tree_{}.html'.format(character)),
                    verbose=True, columns=[character],
                    upload_to_itol=True, tip_size_threshold=25, root_date=1208.8)
