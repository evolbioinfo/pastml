# Execute from the project root:
# $ python -m unittest tests.Pruning_tree_tests -v
#
# Or discover all unittest tests:
# $ python -m unittest tests/*.py -v

import unittest

from ete3 import Tree

from pastml.sb_corrector import PruneMethod, prune_tree_by_state_counts


def add_country_states(tree: Tree, states: dict[str, set[str]]) -> Tree:
    for leaf in tree.iter_leaves():
        leaf.add_feature("Country", states[leaf.name])
    return tree


def make_test_tree() -> Tree:
    """
    Simple two-state tree used for basic pruning-method tests.
    """
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

    return add_country_states(tree, states)


def make_iterative_shortest_tip_tree() -> Tree:
    """
    Tree designed to distinguish static from iterative shortest-tip pruning.

    Initial terminal branches:
        A1 = 0.001
        A2 = 0.0015
        A3 = 0.01
        A4 = 0.02

    If pruning shortest tip branches statically and retaining 2 leaves,
    one would prune A1 and A2.

    If pruning iteratively:
        1. prune A1
        2. A2 becomes terminal with branch 0.0015 + 1.3 = 1.3015
        3. prune A3 next

    Therefore, iterative pruning should retain A2 and A4.
    """
    tree = Tree(
        "((A1:0.001,A2:0.0015):1.3,A3:0.01,A4:0.02);",
        format=1,
    )

    states = {
        "A1": {"A"},
        "A2": {"A"},
        "A3": {"A"},
        "A4": {"A"},
    }

    return add_country_states(tree, states)


def make_iterative_longest_tip_tree() -> Tree:
    """
    Tree designed to distinguish static from iterative longest-tip pruning.

    Initial terminal branches:
        A1 = 10.0
        A2 = 0.1
        A3 = 9.0
        A4 = 8.0

    If pruning longest tip branches statically and retaining 2 leaves,
    one would prune A1 and A3.

    If pruning iteratively:
        1. prune A1
        2. A2 becomes terminal with branch 0.1 + 20.0 = 20.1
        3. prune A2 next

    Therefore, iterative pruning should retain A3 and A4.
    """
    tree = Tree(
        "((A1:10.0,A2:0.1):20.0,A3:9.0,A4:8.0);",
        format=1,
    )

    states = {
        "A1": {"A"},
        "A2": {"A"},
        "A3": {"A"},
        "A4": {"A"},
    }

    return add_country_states(tree, states)


class TestPruneTreeByStateCounts(unittest.TestCase):

    def test_pruning_methods_basic_behavior(self):
        test_cases = [
            (
                PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
                {"A2", "B2"},
            ),
            (
                PruneMethod.PRUNE_LONGEST_TIP_BRANCHES,
                {"A1", "B1"},
            ),
            (
                PruneMethod.PRUNE_SHORTEST_DIST_TO_ROOT,
                {"A1", "B2"},
            ),
            (
                PruneMethod.PRUNE_LONGEST_DIST_TO_ROOT,
                {"A2", "B1"},
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
                    method=method,
                )

                self.assertEqual(
                    set(pruned.get_leaf_names()),
                    expected_leaves,
                )

    def test_prune_shortest_tip_branches_is_iterative(self):
        tree = make_iterative_shortest_tip_tree()

        pruned = prune_tree_by_state_counts(
            root=tree,
            optimized_counts={"A": 2},
            state_column="Country",
            copy_tree=True,
            method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
        )

        self.assertEqual(
            set(pruned.get_leaf_names()),
            {"A2", "A4"},
        )

    def test_prune_longest_tip_branches_is_iterative(self):
        tree = make_iterative_longest_tip_tree()

        pruned = prune_tree_by_state_counts(
            root=tree,
            optimized_counts={"A": 2},
            state_column="Country",
            copy_tree=True,
            method=PruneMethod.PRUNE_LONGEST_TIP_BRANCHES,
        )

        self.assertEqual(
            set(pruned.get_leaf_names()),
            {"A3", "A4"},
        )

    def test_copy_tree_does_not_modify_original(self):
        tree = make_test_tree()
        original_leaves = set(tree.get_leaf_names())

        pruned = prune_tree_by_state_counts(
            root=tree,
            optimized_counts={"A": 1, "B": 1},
            state_column="Country",
            copy_tree=True,
            method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
        )

        self.assertEqual(
            set(tree.get_leaf_names()),
            original_leaves,
        )

        self.assertEqual(
            len(pruned.get_leaf_names()),
            2,
        )

    def test_copy_tree_false_modifies_original(self):
        tree = make_test_tree()

        pruned = prune_tree_by_state_counts(
            root=tree,
            optimized_counts={"A": 1, "B": 1},
            state_column="Country",
            copy_tree=False,
            method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
        )

        self.assertIs(pruned, tree)

        self.assertEqual(
            set(tree.get_leaf_names()),
            {"A2", "B2"},
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
                copy_tree=True,
                method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
            )

    def test_missing_count_for_tree_state(self):
        tree = make_test_tree()

        with self.assertRaisesRegex(
            ValueError,
            r"No optimized count was provided",
        ):
            prune_tree_by_state_counts(
                root=tree,
                optimized_counts={"A": 1},
                state_column="Country",
                copy_tree=True,
                method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
            )

    def test_pruning_preserves_distance_between_retained_leaves(self):
        tree = make_test_tree()

        original_distance = tree.get_distance("A2", "B2")

        pruned = prune_tree_by_state_counts(
            root=tree,
            optimized_counts={"A": 1, "B": 1},
            state_column="Country",
            copy_tree=True,
            method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
        )

        pruned_distance = pruned.get_distance("A2", "B2")

        self.assertAlmostEqual(
            pruned_distance,
            original_distance,
        )

    def test_invalid_pruning_method_raises_value_error(self):
        tree = make_test_tree()

        with self.assertRaisesRegex(
            ValueError,
            r"Unknown pruning method",
        ):
            prune_tree_by_state_counts(
                root=tree,
                optimized_counts={"A": 1, "B": 1},
                state_column="Country",
                copy_tree=True,
                method="NOT_A_REAL_METHOD",
            )


if __name__ == "__main__":
    unittest.main()