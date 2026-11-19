# Execute from the project root:
# $ python -m unittest tests.PruningTreeTest -v
#
# Or discover all unittest tests:
# $ python -m unittest tests/*.py -v

import itertools
import random
import unittest

from ete3 import Tree

import logging

import numpy as np
import pandas as pd

from pastml.sb_corrector import PruneMethod, prune_tree_by_state_counts, optimize_column_counts, \
    bounded_integer_allocation_optimized


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


def get_single_state(leaf) -> str:
    return next(iter(leaf.Country))


def count_states(tree: Tree) -> dict[str, int]:
    counts = {}
    for leaf in tree.iter_leaves():
        state = get_single_state(leaf)
        counts[state] = counts.get(state, 0) + 1
    return counts


def make_random_tree(rng: random.Random, n_tips: int, polytomies: bool) -> Tree:
    """
    Random tree with uniquely named nodes and states drawn from {A, B, C}.

    Tip branch lengths are partly drawn from a small set of values,
    to make sure that ties are present.
    """
    tree = Tree()
    tree.populate(n_tips, random_branches=True)

    if polytomies:
        for node in list(tree.traverse()):
            if not node.is_leaf() and not node.is_root() and rng.random() < 0.3:
                node.delete(preserve_branch_length=True)

    for i, leaf in enumerate(tree.iter_leaves()):
        leaf.name = f"L{i}"
        leaf.dist = round(rng.choice([rng.random(), 0.1, 0.5]), 3)
        leaf.add_feature("Country", {rng.choice("ABC")})

    for i, node in enumerate(tree.traverse()):
        if not node.is_leaf():
            node.name = f"n{i}"

    return tree


def make_random_counts(rng: random.Random, tree: Tree) -> dict[str, int]:
    counts = {
        state: rng.randint(0, count)
        for state, count in count_states(tree).items()
    }
    if not sum(counts.values()):
        counts[next(iter(counts))] = 1
    return counts


def naive_iterative_tip_pruning(tree: Tree, optimized_counts: dict[str, int], prune_shortest: bool) -> Tree:
    """
    Slow reference implementation of the iterative tip-branch pruning:
    at every step, all terminal branches are re-read and the best eligible leaf is pruned.
    Ties are broken by leaf name, as in prune_tree_by_state_counts.
    """
    tree = tree.copy(method="deepcopy")
    current_counts = count_states(tree)

    def key(leaf):
        return (leaf.dist if prune_shortest else -leaf.dist, leaf.name)

    while any(current_counts[state] > optimized_counts[state] for state in current_counts):
        candidates = [
            leaf for leaf in tree.iter_leaves()
            if current_counts[get_single_state(leaf)] > optimized_counts[get_single_state(leaf)]
        ]
        leaf = min(candidates, key=key)
        current_counts[get_single_state(leaf)] -= 1
        leaf.delete(prevent_nondicotomic=True, preserve_branch_length=True)

    return tree


class TestPruneTreeByStateCounts(unittest.TestCase):

    def test_random_trees_against_reference(self):
        """
        Regression test on seeded random trees (with and without polytomies) for all pruning methods:
            - the retained per-state counts match optimized_counts;
            - the distances between retained leaves are preserved;
            - no unary inner node remains (except possibly the root);
            - tip-branch methods retain the same leaves as the naive reference implementation.
        """
        rng = random.Random(2026)
        random.seed(2026)  # Tree.populate uses the global generator

        for i in range(300):
            tree = make_random_tree(rng, n_tips=rng.randint(2, 30), polytomies=bool(i % 2))
            optimized_counts = make_random_counts(rng, tree)

            for method in PruneMethod:
                with self.subTest(tree=i, method=method.value, counts=optimized_counts):
                    pruned, = prune_tree_by_state_counts(
                        forest=tree,
                        optimized_counts=optimized_counts,
                        state_column="Country",
                        copy_trees=True,
                        method=method,
                    )

                    self.assertEqual(
                        count_states(pruned),
                        {state: count for state, count in optimized_counts.items() if count},
                    )

                    for leaf1, leaf2 in itertools.combinations(pruned.get_leaf_names()[:8], 2):
                        self.assertAlmostEqual(
                            pruned.get_distance(leaf1, leaf2),
                            tree.get_distance(leaf1, leaf2),
                        )

                    self.assertFalse(
                        [node.name for node in pruned.traverse()
                         if len(node.children) == 1 and not node.is_root()]
                    )

                    if method in {PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
                                  PruneMethod.PRUNE_LONGEST_TIP_BRANCHES}:
                        reference = naive_iterative_tip_pruning(
                            tree,
                            optimized_counts,
                            prune_shortest=method == PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
                        )
                        self.assertEqual(
                            set(pruned.get_leaf_names()),
                            set(reference.get_leaf_names()),
                        )

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

                pruned, = prune_tree_by_state_counts(
                    forest=tree,
                    optimized_counts={"A": 1, "B": 1},
                    state_column="Country",
                    copy_trees=True,
                    method=method,
                )

                self.assertEqual(
                    set(pruned.get_leaf_names()),
                    expected_leaves,
                )

    def test_prune_shortest_tip_branches_is_iterative(self):
        tree = make_iterative_shortest_tip_tree()

        pruned, = prune_tree_by_state_counts(
            forest=tree,
            optimized_counts={"A": 2},
            state_column="Country",
            copy_trees=True,
            method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
        )

        self.assertEqual(
            set(pruned.get_leaf_names()),
            {"A2", "A4"},
        )

    def test_prune_longest_tip_branches_is_iterative(self):
        tree = make_iterative_longest_tip_tree()

        pruned, = prune_tree_by_state_counts(
            forest=tree,
            optimized_counts={"A": 2},
            state_column="Country",
            copy_trees=True,
            method=PruneMethod.PRUNE_LONGEST_TIP_BRANCHES,
        )

        self.assertEqual(
            set(pruned.get_leaf_names()),
            {"A3", "A4"},
        )

    def test_copy_trees_does_not_modify_original(self):
        tree = make_test_tree()
        original_leaves = set(tree.get_leaf_names())

        pruned, = prune_tree_by_state_counts(
            forest=tree,
            optimized_counts={"A": 1, "B": 1},
            state_column="Country",
            copy_trees=True,
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

    def test_copy_trees_false_modifies_original(self):
        tree = make_test_tree()

        pruned, = prune_tree_by_state_counts(
            forest=tree,
            optimized_counts={"A": 1, "B": 1},
            state_column="Country",
            copy_trees=False,
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
                forest=tree,
                optimized_counts={"A": 3, "B": 1},
                state_column="Country",
                copy_trees=True,
                method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
            )

    def test_missing_count_for_tree_state(self):
        tree = make_test_tree()

        with self.assertRaisesRegex(
            ValueError,
            r"No optimized count was provided",
        ):
            prune_tree_by_state_counts(
                forest=tree,
                optimized_counts={"A": 1},
                state_column="Country",
                copy_trees=True,
                method=PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
            )

    def test_pruning_preserves_distance_between_retained_leaves(self):
        tree = make_test_tree()

        original_distance = tree.get_distance("A2", "B2")

        pruned, = prune_tree_by_state_counts(
            forest=tree,
            optimized_counts={"A": 1, "B": 1},
            state_column="Country",
            copy_trees=True,
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
                forest=tree,
                optimized_counts={"A": 1, "B": 1},
                state_column="Country",
                copy_trees=True,
                method="NOT_A_REAL_METHOD",
            )


def make_forest() -> list[Tree]:
    """
    Forest of 3 trees (A/B/C tips): tree 0 has A3 B2 C1, tree 1 has A2 B2 C1 and an unannotated tip x1,
    tree 2 has A1 B2.
    """
    forest = [
        Tree("((a1:0.3,b1:0.2):0.4,((a2:0.5,c1:0.1):0.2,(b2:0.3,a3:0.6):0.2):0.3);", format=1),
        Tree("((a4:0.2,b3:0.4):0.3,((c2:0.3,a5:0.2):0.1,(b4:0.5,x1:0.1):0.2):0.2);", format=1),
        Tree("((b5:0.3,a6:0.2):0.1,b6:0.4);", format=1),
    ]
    for tree in forest:
        for leaf in tree.iter_leaves():
            if not leaf.name.startswith("x"):
                leaf.add_feature("Country", {leaf.name[0].upper()})
    return forest


def count_tree_states(tree: Tree) -> dict[str, int]:
    counts = {}
    for leaf in tree.iter_leaves():
        states = getattr(leaf, "Country", None)
        if states:
            state = next(iter(states))
            counts[state] = counts.get(state, 0) + 1
    return counts


class TestForestAndMinTipsPerState(unittest.TestCase):
    """
    Forest pruning (states may vanish from a tree, emptied trees are dropped)
    and the minimum number of tips per state over the whole forest (M_MIN_PER_STATE) in the count optimisation.
    """

    def test_state_vanishes_from_a_tree(self):
        """B kept only once: it disappears from some trees, the counts over the forest are met."""
        for method in PruneMethod:
            with self.subTest(method=method.value):
                pruned = prune_tree_by_state_counts(
                    forest=make_forest(),
                    optimized_counts={"A": 6, "B": 1, "C": 2},
                    state_column="Country",
                    copy_trees=True,
                    method=method,
                )
                per_tree = [count_tree_states(tree) for tree in pruned]
                self.assertEqual(sum(c.get("A", 0) for c in per_tree), 6)
                self.assertEqual(sum(c.get("B", 0) for c in per_tree), 1)
                self.assertEqual(sum(c.get("C", 0) for c in per_tree), 2)
                self.assertEqual(len(pruned), 3)

    def test_emptied_trees_are_dropped(self):
        """Very small counts: every returned tree has an annotated tip, and the counts are met."""
        for method in PruneMethod:
            with self.subTest(method=method.value):
                forest = make_forest()
                pruned = prune_tree_by_state_counts(
                    forest=forest,
                    optimized_counts={"A": 1, "B": 1, "C": 1},
                    state_column="Country",
                    copy_trees=True,
                    method=method,
                )
                per_tree = [count_tree_states(tree) for tree in pruned]
                self.assertTrue(all(per_tree))
                self.assertEqual({state: sum(c.get(state, 0) for c in per_tree) for state in "ABC"},
                                 {"A": 1, "B": 1, "C": 1})
                # the original forest is unchanged (copy_trees=True)
                self.assertEqual(len(forest[2].get_leaves()), 3)

    def test_single_tree_emptied_but_unannotated_tip(self):
        """
        The trees whose annotated tips are all pruned are dropped: tree 0 keeps only an unannotated tip,
        tree 2 loses all its tips (for the tip-branch methods, ete3 cannot delete its root).
        """
        forest = [Tree("((a1:0.1,x1:0.2):0.3,a2:0.4);", format=1), Tree("((a3:0.2,b1:0.3):0.1,b2:0.5);", format=1),
                  Tree("((a4:0.1,a5:0.2):0.2,a6:0.3);", format=1)]
        for tree in forest:
            for leaf in tree.iter_leaves():
                if not leaf.name.startswith("x"):
                    leaf.add_feature("Country", {leaf.name[0].upper()})
        for method in PruneMethod:
            with self.subTest(method=method.value):
                # A = 0: tree 0 keeps only its unannotated tip x1, tree 2 is emptied
                pruned = prune_tree_by_state_counts(
                    forest=forest,
                    optimized_counts={"A": 0, "B": 1},
                    state_column="Country",
                    copy_trees=True,
                    method=method,
                )
                self.assertEqual(len(pruned), 1)
                self.assertIn(pruned[0].get_leaf_names(), [["b1"], ["b2"]])

    def test_allocation_lower_bounds(self):
        """The allocation respects the lower bounds, also when they push the start above final_size."""
        available = np.array([20, 5, 5])
        counts = bounded_integer_allocation_optimized(np.array([0.9, 0.05, 0.05]), available, 10,
                                                      lower=np.array([0, 2, 2]))
        self.assertEqual(list(counts), [6, 2, 2])
        # without lower bounds, the old behaviour
        counts = bounded_integer_allocation_optimized(np.array([0.9, 0.05, 0.05]), available, 10)
        self.assertEqual(list(counts), [9, 1, 0])
        # quotas 3/3/0 and lower 0/0/4: the start (3/3/4 = 10) exceeds 6, so items are removed
        counts = bounded_integer_allocation_optimized(np.array([0.5, 0.5, 0.0]), available, 6,
                                                      lower=np.array([0, 0, 4]))
        self.assertEqual(list(counts), [1, 1, 4])
        with self.assertRaises(ValueError):
            bounded_integer_allocation_optimized(np.array([0.5, 0.5, 0.0]), available, 3,
                                                 lower=np.array([2, 2, 0]))

    def _optimize(self, df, M_min=1, M_max=None):
        prior = {state: 1 / 6 for state in "ABCDEF"}
        return optimize_column_counts(column="Country", df=df, prior_distribution=prior, alpha=1.0,
                                      interpolation_method="geometric", _lambda=0.8, M_min=M_min, M_max=M_max)

    def test_optimizer_min_tips_per_state(self):
        """6 states, A over-sampled: every state keeps >= 2 tips; M_min = 10 is raised to 12; M_max = 8 to 12."""
        df = pd.DataFrame({"Country": list("A" * 40 + "BBBCCCDDDEEEFFF")})
        _, counts = self._optimize(df)
        self.assertTrue(all(counts[state] >= 2 for state in "ABCDEF"))
        _, counts = self._optimize(df, M_min=10)
        self.assertGreaterEqual(sum(counts.values()), 12)
        with self.assertLogs("pastml", level=logging.WARNING):
            _, counts = self._optimize(df, M_max=8)
        self.assertEqual(counts, {state: 2 for state in "ABCDEF"})

    def test_optimizer_state_with_prior_zero(self):
        """A state with prior 0 keeps 2 tips (it does not disappear from the subsampled forest)."""
        df = pd.DataFrame({"Country": list("A" * 10 + "B" * 10)})
        _, counts = optimize_column_counts(column="Country", df=df, prior_distribution={"A": 1.0, "B": 0.0},
                                           alpha=1.0, interpolation_method="arithmetic", _lambda=0.99,
                                           M_min=1, M_max=None)
        self.assertEqual(counts["B"], 2)

    def test_optimizer_single_tip_state(self):
        """A state with a single tip keeps it."""
        df = pd.DataFrame({"Country": list("A" * 30 + "B" * 3 + "C")})
        prior = {"A": 0.1, "B": 0.1, "C": 0.8}
        _, counts = optimize_column_counts(column="Country", df=df, prior_distribution=prior, alpha=1.0,
                                           interpolation_method="geometric", _lambda=0.8, M_min=1, M_max=None)
        self.assertEqual(counts["C"], 1)
        self.assertGreaterEqual(counts["B"], 2)

    def test_optimizer_floor_covers_all_tips(self):
        """No pruning when the per-state minimum already retains all the annotated tips."""
        df = pd.DataFrame({"Country": list("AABBCDEF")})
        self.assertIsNone(self._optimize(df)[1])


if __name__ == "__main__":
    unittest.main()