import os
import unittest
from collections import Counter

import numpy as np
import pandas as pd

from pastml.acr import _parse_pastml_parameters
from pastml.annotation import preannotate_forest, get_forest_stats, ForestStats
from pastml.ml import marginal_counts
from pastml.models.JCModel import JCModel
from pastml.models.f81_like import JC
from pastml.tree import read_tree
from pastml.utilities.state_simulator import simulate_states

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.minitree.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')
PARAMS_INPUT = os.path.join(DATA_DIR, 'params.character_Country.method_MPPA.model_JC.tab')


class MRANDJCTest(unittest.TestCase):

    def test_counts(self):

        tree = read_tree(TREE_NWK)
        character = 'Country'
        df = pd.read_csv(STATES_INPUT, index_col=0, header=0)[[character]]
        preannotate_forest([tree], df=df)
        states = np.array([_ for _ in df[character].unique() if not pd.isna(_) and '' != _])
        avg_len, num_nodes, num_tips, tree_len = get_forest_stats([tree])

        freqs, sf, kappa, _ = _parse_pastml_parameters(PARAMS_INPUT, states, num_tips=num_tips, reoptimise=False)
        tau = 0

        model = JCModel(forest_stats=ForestStats([tree]), states=states, parameter_file=PARAMS_INPUT)
        n_repetitions = 50_000
        counts = marginal_counts([tree], character, model, n_repetitions=n_repetitions)

        sim_character = character + '.simulated'
        n_sim_repetitions = n_repetitions * 300
        simulate_states(tree, model, character=sim_character, n_repetitions=n_sim_repetitions)
        good_indices = np.ones(n_sim_repetitions, dtype=int)
        n_states = len(states)
        state2id = dict(zip(states, range(n_states)))
        for tip in tree:
            state_id = state2id[next(iter(getattr(tip, character)))]
            good_indices *= (getattr(tip, sim_character) == state_id).astype(int)
        num_good_simulations = np.count_nonzero(good_indices)
        print('Simulated {} good configurations'.format(num_good_simulations))

        sim_counts = np.zeros((n_states, n_states), dtype=float)
        for n in tree.traverse('levelorder'):
            from_states = getattr(n, sim_character)[good_indices > 0]
            state_nums = Counter(from_states)
            children_transition_counts = Counter()
            for c in n.children:
                transition_counts = Counter(zip(from_states, getattr(c, sim_character)[good_indices > 0]))
                for (i, j), num in transition_counts.items():
                    sim_counts[i, j] += num
                children_transition_counts.update(transition_counts)
            for i, num in state_nums.items():
                sim_counts[i, i] -= min(num, children_transition_counts[(i, i)])
        sim_counts /= num_good_simulations
        print(np.round(counts, 2))
        print(np.round(sim_counts, 2))

        for i in range(n_states):
            for j in range(n_states):
                self.assertAlmostEqual(counts[i, j], sim_counts[i, j], 2,
                                       'Counts are different for {}->{}: {} (calculated) vs {} (simulated).'
                                       .format(states[i], states[j], counts[i, j], sim_counts[i, j]))
