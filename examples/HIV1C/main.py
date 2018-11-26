import logging
import os
import pandas as pd
from ete3 import Tree

from pastml.file import get_combined_ancestral_state_file, get_pastml_parameter_file
from pastml.ml import MPPA, F81
from pastml.acr import pastml_pipeline
from pastml.tree import remove_certain_leaves

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_raxml_tree.nwk')
AFRICAN_TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_raxml_tree_africa.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata_loc.tab')


if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',)
    tree = Tree(TREE_NWK, format=1)
    df = pd.read_table(STATES_INPUT, header=0, index_col=0)
    df.index = df.index.map(str)
    df = df[df.index.isin({_.name for _ in tree})]
    mutations = sorted([_ for _ in df.columns if _.startswith('RT:') or _.startswith('PR:')],
                       key=lambda drm: -len(df[df[drm] == 'resistant']))
    logging.info('3 top mutations are {}'.format(mutations[:3]))

    work_dir = os.path.join(DATA_DIR, 'pastml')
    pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                    html_compressed=os.path.join(DATA_DIR, 'maps', 'map_Loc.html'),
                    verbose=True, columns=['Loc'], work_dir=work_dir,
                    date_column='Year',
                    parameters={'Loc': os.path.join(work_dir, get_pastml_parameter_file(MPPA, F81, 'Loc'))},
                    tip_size_threshold=50)

    column2params = {'Loc': os.path.join(work_dir, get_pastml_parameter_file(MPPA, F81, 'Loc'))}
    for mutation in mutations[:3]:
        column2params[mutation] = os.path.join(work_dir, get_pastml_parameter_file(MPPA, F81, mutation))
        pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                        html_compressed=os.path.join(DATA_DIR, 'maps', 'map_Loc_{}.html'.format(mutation)),
                        verbose=True, columns=['Loc', mutation], name_column='Loc', work_dir=work_dir,
                        date_column='Year', parameters=column2params, tip_size_threshold=20)

    ancestral_states = pd.read_table(os.path.join(work_dir, get_combined_ancestral_state_file(['Loc'])), header=0,
                                     index_col=0)
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
    pastml_pipeline(data=STATES_INPUT, tree=AFRICAN_TREE_NWK,
                    html_compressed=os.path.join(DATA_DIR, 'maps', 'map_Africa_Country.html'),
                    verbose=True, columns=['Country'], work_dir=work_dir,
                    date_column='Year',
                    parameters={'Country': os.path.join(work_dir, get_pastml_parameter_file(MPPA, F81, 'Country'))},
                    tip_size_threshold=25)

    for mutation in mutations[:10]:
        pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                        html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}.html'.format(mutation)),
                        verbose=True, columns=[mutation], work_dir=work_dir,
                        date_column='Year')
