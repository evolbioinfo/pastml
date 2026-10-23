# Execute from the project root:
# $ python -m unittest tests.SBMultiColumnTest -v

import glob
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from ete3 import Tree

import pastml.acr
from pastml.acr import pastml_pipeline
from pastml.ml import MPPA, ALL
from pastml.parsimony import DOWNPASS

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')

# "Trait A" contains a space, to check the column renaming (col_name2cat)
COLUMNS = ['Country', 'Trait A', 'TraitB']
CAT_COLUMNS = ['Country', 'Trait_A', 'TraitB']
RANDOM_SEED = 2026


def write_input_files(tmp_dir):
    """
    Writes the annotation file (the real Country column + two seeded synthetic ones)
    and the prior frequency files (uniform priors over the states of each column).
    """
    df = pd.read_csv(STATES_INPUT, index_col=0, header=0, dtype=str)[['Country']]
    rng = np.random.default_rng(RANDOM_SEED)
    df['Trait A'] = rng.choice(['A1', 'A2', 'A3'], size=len(df), p=[0.6, 0.3, 0.1])
    df['TraitB'] = rng.choice(['B1', 'B2'], size=len(df), p=[0.8, 0.2])
    data = os.path.join(tmp_dir, 'annotations.csv')
    df.to_csv(data)

    priors = {column: {state: 1 / df[column].nunique() for state in sorted(df[column].unique())}
              for column in COLUMNS}
    files = {}

    def write_json(name, columns):
        files[name] = os.path.join(tmp_dir, '{}.json'.format(name))
        with open(files[name], 'w') as f:
            json.dump({column: priors[column] for column in columns}, f)

    write_json('all', COLUMNS)
    write_json('two', COLUMNS[:2])
    write_json('country', COLUMNS[:1])
    files['all_tsv'] = os.path.join(tmp_dir, 'all.tsv')
    pd.DataFrame([(column, state, frequency) for column, s2f in priors.items() for state, frequency in s2f.items()],
                 columns=['Column', 'State', 'Frequency']).to_csv(files['all_tsv'], sep='\t', index=False)
    return data, files


def read_states(work_dir, column):
    """
    Reads the predicted states of the given column: node name -> frozenset of states.
    """
    df = pd.read_csv(os.path.join(work_dir, 'combined_ancestral_states.tab'), sep='\t', index_col=0, dtype=str)
    # in multi-column runs, the combined table also contains rows with empty values for some columns
    return df[column].dropna().groupby(level=0).apply(frozenset)


def read_marginals(work_dir, column):
    path, = glob.glob(os.path.join(work_dir, 'marginal_probabilities.character_{}.*'.format(column)))
    return pd.read_csv(path, sep='\t', index_col=0).sort_index()


class SBMultiColumnTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data, cls.priors = write_input_files(cls.tmp.name)
        cls.work_dirs = {}
        cls.subsampled_states = {}

        # Spy on the states reconstructed on the pruned trees (single process, so that the spy sees them)
        get_inner_node_states = pastml.acr.get_inner_node_states

        def spy(subsampled_forest, character):
            states = get_inner_node_states(subsampled_forest=subsampled_forest, character=character)
            cls.subsampled_states.setdefault(character, []).append(states)
            return states

        with mock.patch.object(pastml.acr, 'get_inner_node_states', spy):
            cls.run_pastml('spy', priors='all', threads=1)
        cls.spy_calls = {character: len(calls) for character, calls in cls.subsampled_states.items()}

        cls.run_pastml('all_json', priors='all', threads=4)
        cls.run_pastml('all_tsv', priors='all_tsv', freq_priors_format='TSV', threads=1)
        cls.run_pastml('two_priors', priors='two', threads=4)
        cls.run_pastml('no_sb', priors=None, threads=4)
        cls.run_pastml('country', priors='country', columns=COLUMNS[:1])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_pastml(cls, name, priors, columns=COLUMNS, threads=1, prediction_method=MPPA, **kwargs):
        work_dir = os.path.join(cls.tmp.name, name)
        pastml_pipeline(tree=TREE_NWK, data=cls.data, data_sep=',', columns=columns,
                        prediction_method=prediction_method,
                        sampling_bias_correction=priors is not None,
                        freq_priors=cls.priors[priors] if priors else None,
                        alpha=1.0, _lambda=0.8, work_dir=work_dir, threads=threads, **kwargs)
        cls.work_dirs[name] = work_dir
        return work_dir

    def assertSameColumn(self, run1, run2, column, column2=None):
        column2 = column2 or column
        self.assertTrue(read_states(self.work_dirs[run1], column).equals(read_states(self.work_dirs[run2], column2)),
                        msg='States of {} differ between {} and {}'.format(column, run1, run2))
        np.testing.assert_allclose(read_marginals(self.work_dirs[run1], column).values,
                                   read_marginals(self.work_dirs[run2], column2).values, atol=1e-10)

    def test_json_vs_tsv_priors(self):
        for column in CAT_COLUMNS:
            with self.subTest(column=column):
                self.assertSameColumn('all_json', 'all_tsv', column)

    def test_processes_vs_single_process(self):
        for column in CAT_COLUMNS:
            with self.subTest(column=column):
                self.assertSameColumn('all_json', 'spy', column)

    def test_columns_are_independent(self):
        for column in CAT_COLUMNS[:2]:
            with self.subTest(column=column):
                self.assertSameColumn('all_json', 'two_priors', column)

    def test_column_without_priors_is_not_corrected(self):
        self.assertSameColumn('two_priors', 'no_sb', 'TraitB')

    def test_single_column_vs_multiple_columns(self):
        self.assertSameColumn('country', 'all_json', 'Country')

    def test_sampling_bias_correction_changes_the_result(self):
        for column in CAT_COLUMNS:
            with self.subTest(column=column):
                self.assertFalse(np.allclose(read_marginals(self.work_dirs['all_json'], column).values,
                                             read_marginals(self.work_dirs['no_sb'], column).values))

    def test_pruned_tree_ran_once_per_column(self):
        self.assertEqual(self.spy_calls, {column: 1 for column in CAT_COLUMNS})

    def test_unambiguous_pruned_tree_states_are_kept(self):
        """
        The inner nodes with a single state reconstructed on the pruned tree keep it on the full tree
        (the ambiguous ones are only restricted to their sets of states).
        """
        for column in CAT_COLUMNS:
            with self.subTest(column=column):
                final_states = read_states(self.work_dirs['spy'], column)
                subsampled_states, = self.subsampled_states[column]
                self.assertTrue(subsampled_states)
                for node, states in subsampled_states.items():
                    if len(states) == 1:
                        self.assertEqual(final_states[node], frozenset(states), msg=node)
                    else:
                        self.assertTrue(final_states[node] <= frozenset(states), msg=node)

    def test_resolve_polytomies(self):
        """
        After the polytomy resolution, the ACR is relaunched with frozen parameters:
        the pruned-tree ACR must not be relaunched (its cached states are reused),
        and the results must not depend on the number of processes.
        """
        # Create polytomies (that PastML can resolve) by collapsing the shorter inner branches
        tree = Tree(TREE_NWK, format=1)
        inner_nodes = [n for n in tree.traverse() if not n.is_leaf() and not n.is_root()]
        threshold = np.percentile([n.dist for n in inner_nodes], 70)
        for n in inner_nodes:
            if n.dist < threshold:
                n.delete(preserve_branch_length=True)
        polytomy_nwk = os.path.join(self.tmp.name, 'polytomies.nwk')
        tree.write(outfile=polytomy_nwk, format=1)

        num_resolved = []
        pruned_acr_calls = []
        resolve_trees, get_inner_node_states = pastml.acr.resolve_trees, pastml.acr.get_inner_node_states

        def resolve_spy(*args, **kwargs):
            num_resolved.append(resolve_trees(*args, **kwargs))
            return num_resolved[-1]

        def get_states_spy(subsampled_forest, character):
            pruned_acr_calls.append(character)
            return get_inner_node_states(subsampled_forest=subsampled_forest, character=character)

        work_dirs = {}
        for threads in (1, 4):
            with mock.patch.object(pastml.acr, 'resolve_trees', resolve_spy), \
                    mock.patch.object(pastml.acr, 'get_inner_node_states', get_states_spy):
                work_dirs[threads] = os.path.join(self.tmp.name, 'polytomies_{}'.format(threads))
                pastml_pipeline(tree=polytomy_nwk, data=self.data, data_sep=',', columns=COLUMNS,
                                sampling_bias_correction=True, freq_priors=self.priors['all'],
                                alpha=1.0, _lambda=0.8, resolve_polytomies=True,
                                work_dir=work_dirs[threads], threads=threads)
            if threads == 1:
                self.assertTrue(num_resolved[0] > 0, msg='No polytomy was resolved, the test tree should be changed')
                self.assertEqual(sorted(pruned_acr_calls), sorted(CAT_COLUMNS))

        for column in CAT_COLUMNS:
            with self.subTest(column=column):
                self.assertTrue(read_states(work_dirs[1], column).equals(read_states(work_dirs[4], column)))
                np.testing.assert_allclose(read_marginals(work_dirs[1], column).values,
                                           read_marginals(work_dirs[4], column).values, atol=1e-10)

    def test_parsimony_on_column_with_priors_raises(self):
        for method in (DOWNPASS, ALL):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'only compatible with the ML methods'):
                    pastml_pipeline(tree=TREE_NWK, data=self.data, data_sep=',', columns=COLUMNS,
                                    prediction_method=[MPPA, method, MPPA], sampling_bias_correction=True,
                                    freq_priors=self.priors['two'],
                                    work_dir=os.path.join(self.tmp.name, 'mp_error'))

    def test_parsimony_on_column_without_priors_is_allowed(self):
        work_dir = self.run_pastml('mp_ok', priors='two', prediction_method=[MPPA, MPPA, DOWNPASS])
        self.assertTrue(len(read_states(work_dir, 'TraitB')))

    def test_missing_priors_file_raises(self):
        with self.assertRaisesRegex(ValueError, 'requires a prior frequencies file'):
            pastml_pipeline(tree=TREE_NWK, data=self.data, data_sep=',', columns=COLUMNS,
                            sampling_bias_correction=True, work_dir=os.path.join(self.tmp.name, 'no_priors'))

    def test_priors_for_unselected_columns_are_ignored(self):
        # PastML only logs warnings in verbose mode
        with self.assertLogs('pastml', level='WARNING') as logs:
            self.run_pastml('unselected', priors='all', columns=COLUMNS[:2], verbose=True)
        self.assertTrue(any('TraitB' in message and 'ignored' in message for message in logs.output))
        self.assertSameColumn('unselected', 'all_json', 'Country')


if __name__ == '__main__':
    unittest.main()
