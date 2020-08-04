import os

from pastml.acr import pastml_pipeline
from pastml.ml import MRAND, MAP
from pastml.models.f81_like import JC

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.subtree.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')
PARAMS_INPUT = os.path.join(DATA_DIR, 'params.character_Country.method_MPPA.model_JC.tab')

if '__main__' == __name__:
    model = JC
    for method in (MRAND, MAP):
        pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK, parameters=PARAMS_INPUT,
                        html_compressed=os.path.join(DATA_DIR, 'maps',
                                                     'Albanian_map_{}_{}.html'.format(method, model)),
                        html=os.path.join(DATA_DIR, 'trees', 'Albanian_tree_{}_{}.html'.format(method, model)),
                        data_sep=',', model=model, verbose=True, prediction_method=method,
                        work_dir=os.path.join(DATA_DIR, 'pastml', method, model))