import os
import unittest
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

from pastml.annotation import preannotate_forest, get_forest_stats
from pastml import get_personalized_feature_name
from pastml.acr import _parse_pastml_parameters
from pastml.ml import optimise_likelihood, initialize_allowed_states, alter_zero_node_allowed_states, \
    get_bottom_up_loglikelihood, calculate_top_down_likelihood, calculate_marginal_likelihoods, \
    convert_likelihoods_to_probabilities, unalter_zero_node_allowed_states, draw_random_states, MRAND, \
    check_marginal_likelihoods, ALLOWED_STATES
from pastml.models.f81_like import JC
from pastml.tree import read_tree

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.subtree.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')
PARAMS_INPUT = os.path.join(DATA_DIR, 'params.character_Country.method_MPPA.model_JC.tab')


class MRANDJCTest(unittest.TestCase):

    def test_selection(self):
        tree = read_tree(TREE_NWK)
        character = 'Country'
        df = pd.read_csv(STATES_INPUT, index_col=0, header=0)[[character]]
        preannotate_forest([tree], df=df)
        states = np.array([_ for _ in df[character].unique() if not pd.isna(_) and '' != _])
        freqs, sf, kappa, _ = _parse_pastml_parameters(PARAMS_INPUT, states, reoptimise=False)
        tree_stats = get_forest_stats([tree])

        tree_len = tree_stats[3]
        num_nodes = tree_stats[1]
        avg_brlen = tree_stats[0]
        model = JC
        likelihood, frequencies, kappa, sf, tau = \
            optimise_likelihood(forest=[tree], avg_br_len=avg_brlen, num_edges=num_nodes - 1, tree_len=tree_len,
                                character=character, states=states, model=model,
                                frequencies=freqs, observed_frequencies=freqs, kappa=kappa, sf=sf,
                                tau=0, optimise_frequencies=False,
                                optimise_kappa=False, optimise_sf=False, optimise_tau=False)

        initialize_allowed_states(tree, character, states)
        altered_nodes = []
        if 0 == tau:
            altered_nodes = alter_zero_node_allowed_states(tree, character)
        get_bottom_up_loglikelihood(tree=tree, character=character, frequencies=frequencies, sf=sf, kappa=kappa,
                                    is_marginal=True, model=model, tau=tau,
                                    tree_len=tree_len, num_edges=num_nodes - 1, alter=False)
        calculate_top_down_likelihood(tree, character, frequencies, sf, tree_len=tree_len, num_edges=num_nodes - 1,
                                      kappa=kappa, model=model, tau=tau)
        calculate_marginal_likelihoods(tree, character, frequencies)
        check_marginal_likelihoods(tree, character)
        mp_df = convert_likelihoods_to_probabilities(tree, character, states)
        if altered_nodes:
            unalter_zero_node_allowed_states(altered_nodes, character)

        def work(i):
            feature_i = '{}_{}.{}'.format(character, MRAND, i)
            for n in tree.traverse():
                n.add_feature(feature_i, getattr(n, character, set()))
            draw_random_states([tree], feature_i, model, states, frequencies, sf, kappa, tau, tree_len, num_nodes)

        n_repetitions = 100_000
        with ThreadPool() as pool:
            pool.map(func=work, iterable=range(n_repetitions))

        features = [get_personalized_feature_name('{}_{}.{}'.format(character, MRAND, i), ALLOWED_STATES)
                    for i in range(n_repetitions)]
        for n in tree.traverse():
            mps = mp_df.loc[n.name, :]
            prob = np.zeros(len(mps), dtype=np.float64)
            for feature in features:
                prob += getattr(n, feature)
            prob /= prob.sum()
            print(n.name, np.round(list(mps), 2), np.round(prob, 2))
            for mp, p in zip(mps, prob):
                self.assertAlmostEqual(mp, p, 2,
                                       msg='Randomly drawn states ({}) do not seem to follow '
                                           'marginal probabilities ({}) for {}.'.format(list(mps), prob, n.name))
