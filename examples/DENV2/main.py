import os

from ml import ML
from pastml.acr import pastml_pipeline

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'real', 'raxml_tree.dated.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'real', 'metadata.tab')

if '__main__' == __name__:
    character = 'Location'
    pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK, prediction_method=ML,
                    html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}.html'.format(character)),
                    verbose=True, columns=[character], date_column='year',
                    upload_to_itol=True, itol_tree_name='Location')
