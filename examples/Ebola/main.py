import os

from pastml.acr import pastml_pipeline, get_pastml_parameter_file
from pastml.ml import MPPA, F81, JC, JOINT, MAP

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Makona_1610_cds_ig.MCC.tree.nwk')
STATES_CSV = os.path.join(DATA_DIR, 'Makona_1610_metadata_2016-06-23.csv')


if '__main__' == __name__:
    loc = 'location'
    for model in (F81, JC):
        parameter_file = os.path.join(DATA_DIR, 'pastml_files',
                                      'pastml_{}_{}'.format(MPPA, model), get_pastml_parameter_file(MPPA, model, loc))
        for method in (MPPA, JOINT, MAP):
            work_dir = os.path.join(DATA_DIR, 'pastml_files', 'pastml_{}_{}'.format(method, model))
            pastml_pipeline(data=STATES_CSV, tree=TREE_NWK, data_sep=',',
                            html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}_{}.html'.format(method, model)),
                            html=os.path.join(DATA_DIR, 'trees', 'tree_{}_{}.html'.format(method, model)),
                            prediction_method=method, model=model,
                            verbose=True, columns=[loc], work_dir=work_dir,
                            column2parameters={loc: parameter_file} if os.path.exists(parameter_file) else None)
