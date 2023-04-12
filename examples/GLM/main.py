import os

from pastml.acr import pastml_pipeline
from pastml.ml import MPPA
from pastml.models.GLMModel import GLM, GLM_MATRICES, GLM_COEFFICIENTS

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')

method = MPPA
model = GLM

# Write two input matrices into the parameter file:
param_file = os.path.join(DATA_DIR, 'glm_params.tab')
with open(param_file, 'w+') as f:
    f.write('parameter\tvalue\n')
    f.write('{}\t{}\n'.format(GLM_MATRICES, '; '.join((os.path.join(DATA_DIR, 'm1.txt'),
                                                     os.path.join(DATA_DIR, 'm2.txt')))))

pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                html_compressed=os.path.join(DATA_DIR, 'pastml_optimized_coeffs',
                                             'Albanian_map_{}_{}.html'.format(method, model)),
                html=os.path.join(DATA_DIR, 'pastml_optimized_coeffs',
                                  'Albanian_tree_{}_{}.html'.format(method, model)),
                data_sep=',', model=model, verbose=True, prediction_method=method,
                work_dir=os.path.join(DATA_DIR, 'pastml_optimized_coeffs'),
                parameters=param_file)

# Write two input matrices into the parameter file, and also fixed coefficients [0.5 and 0.1]:
param_file_coeffs = os.path.join(DATA_DIR, 'glm_params_coeffs.tab')
with open(param_file_coeffs, 'w+') as f:
    f.write('parameter\tvalue\n')
    f.write('{}\t{}\n'.format(GLM_MATRICES, '; '.join((os.path.join(DATA_DIR, 'm1.txt'),
                                                       os.path.join(DATA_DIR, 'm2.txt')))))
    f.write('{}\t{}\n'.format(GLM_COEFFICIENTS, '0.5; 0.1'))

pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                html_compressed=os.path.join(DATA_DIR, 'pastml_fixed_coeffs',
                                             'Albanian_map_{}_{}_fixed.html'.format(method, model)),
                html=os.path.join(DATA_DIR, 'pastml_optimized_coeffs',
                                  'Albanian_tree_{}_{}.html'.format(method, model)),
                data_sep=',', model=model, verbose=True, prediction_method=method,
                work_dir=os.path.join(DATA_DIR, 'pastml_optimized_coeffs'),
                parameters=param_file_coeffs)
