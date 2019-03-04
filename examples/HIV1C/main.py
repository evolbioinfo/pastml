import logging
import os

from pastml.acr import pastml_pipeline

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_phyml_tree.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata.tab')


if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',)

    mutations = ['RT:M184V', 'RT:K103N']
    pastml_pipeline(data=STATES_INPUT,
                    tree=TREE_NWK,
                    html_compressed=os.path.join(DATA_DIR, 'maps', 'map_{}.html'.format('_'.join(mutations))),
                    verbose=True, columns=mutations, work_dir=os.path.join(DATA_DIR, 'pastml'),
                    tip_size_threshold=12, date_column='Year', upload_to_itol=True)
