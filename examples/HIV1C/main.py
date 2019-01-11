import logging
import os
import random

import pandas as pd
from ete3 import Tree

from pastml import get_personalized_feature_name
from pastml.acr import pastml_pipeline, COPY
from pastml.file import get_pastml_parameter_file, get_combined_ancestral_state_file
from pastml.visualisation.generate_geomap import generate_map
from pastml.ml import MPPA, F81, ML
from tree import remove_certain_leaves

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_{}_tree.nwk')
AFRICAN_TREE_NWK = os.path.join(DATA_DIR, 'best', 'pastml_{}_tree_africa.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'metadata_loc.tab')

if '__main__' == __name__:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', )

    model = F81
    mutations = ['RT:M184V', 'RT:K103N', 'RT:D67N']
    character = 'Loc'
    method = MPPA

    # # All the trees have the same tips
    # nwk = TREE_NWK.format('raxml')
    # tree = Tree(nwk, format=1)
    # tips = {_.name for _ in tree}
    #
    # df = pd.read_table(STATES_INPUT, header=0, index_col=0)
    # df.index = df.index.map(str)
    # df = df.loc[tips, ['Loc']]
    # locations = [_ for _ in df['Loc'].unique() if not pd.isnull(_) and _ != '']
    # fixed_tips = []
    # for loc in locations:
    #     fixed_tips.extend(random.sample(set(df[df['Loc'] == loc].index), 10))
    # fixed_tips = set(fixed_tips)
    # print(fixed_tips)
    #
    # # for tree_type in ('phyml', 'raxml', 'fast'):
    # for tree_type in ('raxml',):
    #     nwk = TREE_NWK.format(tree_type)
    #     tree = Tree(nwk, format=1)
    #     tips = {_.name for _ in tree if _.name not in fixed_tips}
    #
    #     df = None
    #     for num in (1000, 500, 250, 125):
    #         logging.info('{} tree with {} tips.'.format(tree_type, num))
    #         tips = set(random.sample(tips, num - len(fixed_tips)))
    #         tree = remove_certain_leaves(tree, to_remove=lambda node: node.name not in tips | fixed_tips)
    #         nwk = '{}_tips_{}.nwk'.format(TREE_NWK.format(tree_type), num)
    #         tree.write(outfile=nwk, format=3, format_root_node=True)
    #         work_dir = os.path.join(DATA_DIR, 'pastml', model, method, character, tree_type, 'tips_{}'.format(num))
    #         map = os.path.join(DATA_DIR, 'maps', 'map_{}_{}_{}_{}_{}.html'.format(character, model, method, tree_type,
    #                                                                               'tips_{}'.format(num)))
    #         html = os.path.join(DATA_DIR, 'trees',
    #                             'tree_{}_{}_{}_{}_{}.html'.format(character, model, method, tree_type,
    #                                                               'tips_{}'.format(num)))
    #
    #         pastml_pipeline(data=STATES_INPUT, tree=nwk,
    #                         html_compressed=map, html=html,
    #                         model=model, verbose=True, prediction_method=method, columns=character,
    #                         work_dir=work_dir, tip_size_threshold=20, date_column='Year',
    #                         # parameters=os.path.join(work_dir, get_pastml_parameter_file(method, model, character))
    #                         )
    #
    #         state_df = pd.read_table(os.path.join(work_dir, get_combined_ancestral_state_file([character])),
    #                                  header=0, index_col=0)
    #         state_df.columns = ['{}_{}'.format(character, num)]
    #         if df is None:
    #             df = state_df
    #         else:
    #             df = df.join(state_df)
    #
    #     combined_tab = os.path.join(DATA_DIR, 'pastml', model, method, character, tree_type, 'states_by_years.tab')
    #     df.to_csv(combined_tab, sep='\t')
    #
    #     for num in (1000, 500, 250, 125):
    #         work_dir = os.path.join(DATA_DIR, 'pastml', model, method, character, tree_type, 'tips_combined_{}'.format(num))
    #         map = os.path.join(DATA_DIR, 'maps', 'map_{}_{}_{}_{}_{}.html'.format(character, model, method, tree_type,
    #                                                                               'tips_combined_{}'.format(num)))
    #         pastml_pipeline(data=combined_tab, tree='{}_tips_{}.nwk'.format(TREE_NWK.format(tree_type), num),
    #                         html_compressed=map,
    #                         model=model, verbose=True, prediction_method=COPY,
    #                         work_dir=work_dir, tip_size_threshold=20)

    for tree_type in ('raxml', 'phyml', 'fast'):
        column2parameter = {}
        nwk = TREE_NWK.format(tree_type)

        generate_map(data=STATES_INPUT, country='Country', location='Loc',
                     tree=nwk, html=os.path.join(DATA_DIR, 'maps', 'geomap_{}.html'.format(tree_type)))

        character = 'Loc'
        param_file = os.path.join(DATA_DIR, 'pastml', model, ML, tree_type, character,
                                  get_pastml_parameter_file(MPPA, model,
                                                            get_personalized_feature_name(character, MPPA)))
        for method in (ML, MPPA):
            if os.path.exists(param_file):
                column2parameter[character] = param_file
            pastml_pipeline(data=STATES_INPUT, tree=nwk,
                            html_compressed=os.path.join(DATA_DIR, 'maps',
                                                         'map_{}_{}_{}_{}.html'
                                                         .format(character, model, method, tree_type)),
                            model=model, verbose=True, prediction_method=method, columns=character, date_column='Year',
                            work_dir=os.path.join(DATA_DIR, 'pastml', model, method, tree_type, character),
                            parameters=column2parameter, tip_size_threshold=20)

        for character in mutations:
                param_file = os.path.join(DATA_DIR, 'pastml', model, ML, tree_type, character,
                                          get_pastml_parameter_file(MPPA, model,
                                                                    get_personalized_feature_name(character, MPPA)))
                for method in (ML, MPPA):
                    if os.path.exists(param_file):
                        column2parameter[character] = param_file
                    pastml_pipeline(data=STATES_INPUT, tree=nwk,
                                    html_compressed=os.path.join(DATA_DIR, 'maps',
                                                                 'map_{}_{}_{}_{}.html'
                                                                 .format(character, model, method, tree_type)),
                                    model=model, verbose=True, prediction_method=method, columns=character,
                                    date_column='Year',
                                    work_dir=os.path.join(DATA_DIR, 'pastml', model, method, tree_type, character),
                                    parameters=column2parameter, tip_size_threshold=20)

        for mutation in mutations:
            for method in (ML, MPPA):
                pastml_pipeline(data=STATES_INPUT, tree=nwk,
                                html_compressed=os.path.join(DATA_DIR, 'maps',
                                                             'map_Loc_{}_{}_{}_{}.html'
                                                             .format(mutation, model, method, tree_type)),
                                model=model, verbose=True, prediction_method=method, columns=['Loc', mutation],
                                name_column='Loc', date_column='Year',
                                work_dir=os.path.join(DATA_DIR, 'pastml', model, method, tree_type,
                                                      'Loc_{}'.format(mutation)),
                                parameters=column2parameter, tip_size_threshold=20)

    # ancestral_states = pd.read_table(os.path.join(DATA_DIR, 'pastml', model, MPPA, tree_type, 'Loc',
    #                                               get_combined_ancestral_state_file(['Loc'])), header=0, index_col=0)
    # tree = Tree(nwk, format=1)
    # todo = [tree]
    # while todo:
    #     node = todo.pop()
    #     if 'Africa' in ancestral_states.loc[node.name, 'Loc']:
    #         if node.is_leaf():
    #             node.add_feature('keep', True)
    #         else:
    #             todo.extend(node.children)
    # tree = remove_certain_leaves(tree, to_remove=lambda node: not getattr(node, 'keep', False))
    # african_nwk = AFRICAN_TREE_NWK.format(tree_type)
    # tree.write(outfile=african_nwk, format=3)
    #
    # for method in (ML, MPPA):
    #     param_file = os.path.join(DATA_DIR, 'pastml', model, ML, tree_type, 'Country',
    #                               get_pastml_parameter_file(MPPA, model,
    #                                                         get_personalized_feature_name('Country', MPPA)))
    #     pastml_pipeline(data=STATES_INPUT, tree=african_nwk,
    #                     html_compressed=os.path.join(DATA_DIR, 'maps',
    #                                                  'map_Africa_{}_{}_{}_{}.html'
    #                                                  .format('Country', model, method, tree_type)),
    #                     model=model, verbose=True, prediction_method=method, columns='Country',
    #                     work_dir=os.path.join(DATA_DIR, 'pastml', model, method, tree_type, 'Country'),
    #                     parameters=param_file if os.path.exists(param_file) else None, tip_size_threshold=20)
