import unittest

from ete3 import Tree

from pastml.annotation import ForestStats
from pastml.tree import annotate_dates

tree1 = Tree('(A:1,(B:1,C:1.5)BC:2)r1:0;', format=3)
tree2 = Tree('(D:1,E:1.5)r2:0;', format=3)
tree3 = Tree('(F:1,G:2)r3:0;', format=3)
tree4 = Tree('(H:1,I:1.5,J:1)r4:1;', format=3)

forest = [tree1, tree2, tree3, tree4]
annotate_dates(forest, root_dates=[1902, 1900, 1903, 1905])

annotations = {'A': {'Africa'},
               'B': {'Africa', 'Europe'},
               'C': set(),
               'D': {'Asia'},
               'E': {'Asia', 'Africa', 'Europe'},
               'F': {'Africa'},
               'G': {'China'},
               'H': set(),
               'J': {'France'},
               'I': {'UK', 'France'}}

for tree in forest:
    for n in tree:
        n.add_feature('loc', annotations[n.name])


class ForestStatTest(unittest.TestCase):
    def test_stat_tree_dates(self):
        fs = ForestStats(forest)
        self.assertEqual(1900, fs.min_tree_date)
        self.assertEqual(1906.5, fs.max_tree_date)

    def test_stat_tree_dates_root_branch(self):
        fs = ForestStats([tree4])
        self.assertEqual(1904, fs.min_tree_date)
        self.assertEqual(1906.5, fs.max_tree_date)

    def test_stat_tree_dates_interval(self):
        fs = ForestStats(forest, min_date=1901, max_date=1906)
        self.assertEqual(1901, fs.min_tree_date)
        self.assertEqual(1906, fs.max_tree_date)

    def test_interval_dates_interval(self):
        fs = ForestStats(forest, min_date=1901, max_date=1906)
        self.assertEqual(fs.min_interval_date, 1901)
        self.assertEqual(fs.max_interval_date, 1906)

    def test_within_bounds(self):
        fs = ForestStats(forest, min_date=1901, max_date=1903)
        self.assertFalse(fs.node_within_bounds(tree2))
        self.assertFalse(fs.node_within_bounds(tree4.children[0]))
        self.assertFalse(fs.node_within_bounds(tree1.children[1]))
        self.assertTrue(fs.node_within_bounds(tree2.children[0]))
        self.assertTrue(fs.node_within_bounds(tree1))

        self.assertFalse(fs.tree_within_bounds(tree4))
        self.assertTrue(fs.tree_within_bounds(tree1))
        self.assertTrue(fs.tree_within_bounds(tree2))

        self.assertTrue(fs.branch_within_bounds(tree1.children[0]))
        self.assertTrue(fs.branch_within_bounds(tree2.children[1]))
        self.assertFalse(fs.branch_within_bounds(tree2))
        self.assertFalse(fs.branch_within_bounds(tree3.children[0]))
        self.assertFalse(fs.branch_within_bounds(tree4.children[0]))

        fs = ForestStats(forest, min_date=1901, max_date=1904.5)
        self.assertTrue(fs.tree_within_bounds(tree4))

    def test_n(self):
        fs = ForestStats(forest)
        self.assertEqual(10, fs.n_tips)
        self.assertEqual(15, fs.n_nodes)
        self.assertEqual(4, fs.n_trees)
        self.assertEqual(3, fs.n_zero_nodes)
        self.assertEqual(0, fs.n_zero_tips)

    def test_n_interval(self):
        fs = ForestStats(forest, min_date=1901, max_date=1903)
        self.assertEqual(4, fs.n_tips)
        self.assertEqual(5, fs.n_nodes)
        self.assertEqual(3, fs.n_trees)
        self.assertEqual(2, fs.n_zero_nodes)
        self.assertEqual(1, fs.n_zero_tips)

    def test_length(self):
        fs = ForestStats(forest)
        self.assertEqual(15.5, fs.length)
        fs = ForestStats(forest, min_date=1901, max_date=1903)
        self.assertEqual(2.5, fs.length)
        fs = ForestStats(forest, min_date=1903, max_date=1904.5)
        self.assertEqual(5, fs.length)

    def test_avg_brlen(self):
        fs = ForestStats([tree1, tree2])
        self.assertEqual(8 / 6, fs.avg_dist)
        fs = ForestStats([tree1, tree2], min_date=1901, max_date=1903)
        self.assertEqual(2.5 / 3, fs.avg_dist)

    def test_polytomies(self):
        fs = ForestStats(forest)
        self.assertEqual(1, fs.n_polytomies)
        fs = ForestStats(forest, min_date=1901, max_date=1905)
        self.assertEqual(0, fs.n_polytomies)
        fs = ForestStats(forest, min_date=1901, max_date=1905.5)
        self.assertEqual(1, fs.n_polytomies)

    def test_missing_data(self):
        fs = ForestStats(forest, 'loc')
        self.assertEqual(0.2, fs.missing_data)
        fs = ForestStats(forest, 'loc', max_date=1904.5)
        self.assertEqual(0, fs.missing_data)
        fs = ForestStats(forest, 'loc', max_date=1906)
        self.assertEqual(1/7, fs.missing_data)

    def test_observed_frequencies(self):
        fs = ForestStats(forest, 'loc')
        self.assertAlmostEqual((2.5 + 1/3)/8, fs.observed_frequencies['Africa'], places=3)
        self.assertAlmostEqual(1/8, fs.observed_frequencies['China'], places=3)
        self.assertAlmostEqual(0.5/8, fs.observed_frequencies['UK'], places=3)
        self.assertAlmostEqual(1.5/8, fs.observed_frequencies['France'], places=3)
        self.assertAlmostEqual(0, fs.observed_frequencies['Italy'], places=3)

    def test_observed_frequencies_interval(self):
        fs = ForestStats(forest, 'loc', min_date=1901.1, max_date=1904)
        self.assertAlmostEqual((1 + 1/3)/2, fs.observed_frequencies['Africa'], places=3)
        self.assertAlmostEqual(1/6, fs.observed_frequencies['Asia'], places=3)
        self.assertAlmostEqual(0, fs.observed_frequencies['UK'], places=3)
