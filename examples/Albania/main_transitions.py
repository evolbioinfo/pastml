import os

from pastml.acr import pastml_pipeline, COPY
from pastml.ml import MPPA, MAP, JOINT, ALL, ML
from pastml.models.f81_like import EFT, F81, JC
from pastml.parsimony import ACCTRAN, DELTRAN, DOWNPASS, MP
from pastml.file import get_pastml_parameter_file
from utilities.transition_counter import count_transitions

COL = 'Country'

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')
STATES_COPY = os.path.join(DATA_DIR, 'copy_states.csv')

if '__main__' == __name__:
    count_transitions(data=STATES_INPUT, tree=TREE_NWK, column=COL,
                      html=os.path.join(DATA_DIR, 'transitions_Albanian_tree_{}.html'.format(F81)),
                      out_transitions=os.path.join(DATA_DIR, 'transitions_Albanian_tree_{}.tab'.format(F81)),
                      data_sep=',', verbose=True, threshold=1,
                      work_dir=os.path.join(DATA_DIR, 'pastml', 'transitions'),
                      n_repetitions=1000,
                      parameters=os.path.join(DATA_DIR, 'pastml', MPPA, F81, get_pastml_parameter_file(MPPA, F81, COL)))