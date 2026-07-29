# Execute from the project root:
# $ python -m unittest tests.Pruning_tree_tests -v
#
# Or discover all unittest tests:
# $ python -m unittest tests/*.py -v

import unittest

from ete3 import Tree

from pastml.sb_corrector import PruneMethod, prune_tree_by_state_counts


def make_test_tree() -> Tree:
    tree = Tree(
        "((A1:0.1,B2:0.6):2.0,(A2:0.8,B1:0.2):0.1);",
        format=1,
    )

    states = {
        "A1": {"A"},
        "A2": {"A"},
        "B1": {"B"},
        "B2": {"B"},
    }

    for leaf in tree.iter_leaves():
        leaf.add_feature("Country", states[leaf.name])

    return tree


class TestPruneTreeByStateCounts(unittest.TestCase):

    def test_pruning_methods(self):
        test_cases = [
            (
                PruneMethod.RETAIN_SHORTEST_TIP_BRANCHES,
                {"A1", "B1"}
            ),
            (
                PruneMethod.RETAIN_LONGEST_TIP_BRANCHES,
                {"A2", "B2"}
            ),
            (
                PruneMethod.RETAIN_SHORTEST_DIST_TO_ROOT,
                {"A2", "B1"}
            ),
            (
                PruneMethod.RETAIN_LONGEST_DIST_TO_ROOT,
                {"A1", "B2"}
            ),
        ]

        for method, expected_leaves in test_cases:
            with self.subTest(method=method):
                tree = make_test_tree()

                pruned = prune_tree_by_state_counts(
                    root=tree,
                    optimized_counts={"A": 1, "B": 1},
                    state_column="Country",
                    copy_tree=True,
                    method=method
                )

                self.assertEqual(
                    set(pruned.get_leaf_names()),
                    expected_leaves
                )

    def test_copy_tree_does_not_modify_original(self):
        tree = make_test_tree()
        original_leaves = set(tree.get_leaf_names())

        pruned = prune_tree_by_state_counts(
            root=tree,
            optimized_counts={"A": 1, "B": 1},
            state_column="Country",
            copy_tree=True
        )

        self.assertEqual(
            set(tree.get_leaf_names()),
            original_leaves
        )

        self.assertEqual(
            len(pruned.get_leaf_names()),
            2
        )

    def test_requested_count_exceeds_available(self):
        tree = make_test_tree()

        with self.assertRaisesRegex(
            ValueError,
            r"Cannot retain 3 leaves for state 'A'",
        ):
            prune_tree_by_state_counts(
                root=tree,
                optimized_counts={"A": 3, "B": 1},
                state_column="Country",
                copy_tree=True
            )

    def test_pruning_preserves_distance_between_retained_leaves(self):
        tree = make_test_tree()

        original_distance = tree.get_distance("A1", "B1")

        pruned = prune_tree_by_state_counts(
            root=tree,
            optimized_counts={"A": 1, "B": 1},
            state_column="Country",
            copy_tree=True,
            method=PruneMethod.RETAIN_SHORTEST_TIP_BRANCHES
        )

        pruned_distance = pruned.get_distance("A1", "B1")

        self.assertAlmostEqual(
            pruned_distance,
            original_distance
        )
