import logging
import os
import pandas as pd
from ete3 import Tree

from pastml.acr import pastml_pipeline

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_raxml_tree.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata_loc.tab')


if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',)
    tree = Tree(TREE_NWK, format=1)
    df = pd.read_table(STATES_INPUT, header=0, index_col=0)
    df = df[df.index.map(str).isin({_.name for _ in tree})]
    mutations = sorted([_ for _ in df.columns if _.startswith('RT:') or _.startswith('PR:')],
                       key=lambda drm: -len(df[df[drm] == 'resistant']))
    logging.info('3 top mutations are {}'.format(mutations[:3]))

    pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                    html_compressed=os.path.join(DATA_DIR, 'maps', 'map_Loc.html'),
                    verbose=True, columns=['Loc'], work_dir=os.path.join(DATA_DIR, 'pastml'),
                    date_column='Year')

    for mutation in mutations[:10]:
        pastml_pipeline(data=STATES_INPUT, tree=TREE_NWK,
                        html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}.html'.format(mutation)),
                        verbose=True, columns=[mutation], work_dir=os.path.join(DATA_DIR, 'pastml'),
                        date_column='Year')
