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
TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_raxml_tree.nwk')
AFRICAN_TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_raxml_tree_africa.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata_loc.tab')


if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',)

    model = F81

    mutations = ['RT:M184V', 'RT:K103N', 'RT:D67N']
    column2parameter = {}
    for character in ['Loc'] + mutations:
            param_file = os.path.join(DATA_DIR, 'pastml', model, ML, character,
                                      get_pastml_parameter_file(MPPA, model,
                                                                get_personalized_feature_name(character, MPPA)))
            if os.path.exists(param_file):
                column2parameter[character] = param_file
            for method in (ML, MPPA):
                pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                                html_compressed=os.path.join(DATA_DIR, 'maps',
                                                             'map_{}_{}_{}.html'.format(character, model, method)),
                                model=model, verbose=True, prediction_method=method, columns=character,
                                work_dir=os.path.join(DATA_DIR, 'pastml', model, method, character),
                                parameters=column2parameter, tip_size_threshold=20)
    for mutation in mutations:
        for method in (ML, MPPA):
            pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                            html_compressed=os.path.join(DATA_DIR, 'maps',
                                                         'map_Loc_{}_{}_{}.html'.format(mutation, model, method)),
                            model=model, verbose=True, prediction_method=method, columns=['Loc', mutation],
                            name_column='Loc',
                            work_dir=os.path.join(DATA_DIR, 'pastml', model, method, 'Loc_{}'.format(mutation)),
                            parameters=column2parameter, tip_size_threshold=20)

    ancestral_states = pd.read_table(os.path.join(DATA_DIR, 'pastml', model, MPPA, 'Loc',
                                                  get_combined_ancestral_state_file(['Loc'])), header=0, index_col=0)
    tree = Tree(TREE_NWK, format=1)
    todo = [tree]
    while todo:
        node = todo.pop()
        if 'Africa' in ancestral_states.loc[node.name, 'Loc']:
            if node.is_leaf():
                node.add_feature('keep', True)
            else:
                todo.extend(node.children)
    tree = remove_certain_leaves(tree, to_remove=lambda node: not getattr(node, 'keep', False))
    tree.write(outfile=AFRICAN_TREE_NWK, format=3)

    for method in (ML, MPPA):
        param_file = os.path.join(DATA_DIR, 'pastml', model, ML, 'Country',
                                  get_pastml_parameter_file(MPPA, model,
                                                            get_personalized_feature_name('Country', MPPA)))
        pastml_pipeline(data=STATES_INPUT, tree=AFRICAN_TREE_NWK,
                        html_compressed=os.path.join(DATA_DIR, 'maps',
                                                     'map_Africa_{}_{}_{}.html'.format('Country', model, method)),
                        model=model, verbose=True, prediction_method=method, columns='Country',
                        work_dir=os.path.join(DATA_DIR, 'pastml', model, method, 'Country'),
                        parameters=param_file if os.path.exists(param_file) else None, tip_size_threshold=20)
