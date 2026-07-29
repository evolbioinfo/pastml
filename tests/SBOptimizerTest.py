# Execute from the project root:
# $ python -m unittest tests.SBOptimizerTest -v
#
# Or discover all unittest tests:
# $ python -m unittest tests/*.py -v

import unittest

import numpy as np

from pastml.sb_corrector import (
    optimize_counts,
    optimize_counts_optimized
)


NUMBER_OF_RANDOM_SCENARIOS = 1_000
RANDOM_SEED = 42

# Keep random datasets relatively small because optimize_counts() is the
# intentionally slower reference implementation.
MIN_NUMBER_OF_STATES = 2
MAX_NUMBER_OF_STATES = 8
MIN_AVAILABLE_PER_STATE = 1
MAX_AVAILABLE_PER_STATE = 15

OBJECTIVE_TOLERANCE = 1e-12

def assert_feasible_allocation(
    test_case: unittest.TestCase,
    counts: dict[str, int],
    observed_counts: dict[str, int],
) -> None:
    """
    Check that an optimizer result satisfies all allocation constraints.
    """
    test_case.assertEqual(
        set(counts),
        set(observed_counts),
        msg="Returned labels differ from the observed-count labels.",
    )

    for label, count in counts.items():
        test_case.assertIsInstance(
            count,
            int,
            msg=f"Count for '{label}' is not an integer: {count!r}.",
        )

        test_case.assertGreaterEqual(
            count,
            0,
            msg=f"Count for '{label}' is negative.",
        )

        test_case.assertLessEqual(
            count,
            observed_counts[label],
            msg=(
                f"Count for '{label}' exceeds availability: "
                f"{count} > {observed_counts[label]}."
            ),
        )

    test_case.assertGreater(
        sum(counts.values()),
        0,
        msg="The optimizer returned an empty final dataset.",
    )

    test_case.assertLessEqual(
        sum(counts.values()),
        sum(observed_counts.values()),
        msg="The final size exceeds the initial dataset size.",
    )


class TestOptimizedCountOptimizer(unittest.TestCase):

    def test_random_scenarios_have_equal_objective_values(self):
        rng = np.random.default_rng(RANDOM_SEED)

        exact_count_matches = 0
        tied_count_differences = 0

        for scenario_id in range(NUMBER_OF_RANDOM_SCENARIOS):
            number_of_states = int(
                rng.integers(
                    MIN_NUMBER_OF_STATES,
                    MAX_NUMBER_OF_STATES + 1,
                )
            )

            labels = [
                f"state_{index}"
                for index in range(number_of_states)
            ]

            available_array = rng.integers(
                MIN_AVAILABLE_PER_STATE,
                MAX_AVAILABLE_PER_STATE + 1,
                size=number_of_states,
            )

            observed_counts = {
                label: int(count)
                for label, count in zip(labels, available_array)
            }

            # Dirichlet generates a non-negative probability vector
            # whose entries sum to one.
            target_array = rng.dirichlet(
                np.ones(number_of_states)
            )

            target_distribution = {
                label: float(probability)
                for label, probability in zip(labels, target_array)
            }

            optimizer_lambda = float(rng.uniform(0.0, 1.0))

            with self.subTest(
                scenario=scenario_id,
                number_of_states=number_of_states,
                initial_size=sum(observed_counts.values()),
                optimizer_lambda=optimizer_lambda,
            ):
                # The standard optimizer currently prints the objective for
                # every tested final size. Suppress that output during tests.
                standard_counts, standard_objective = \
                    optimize_counts(
                        target_distribution=target_distribution,
                        observed_counts=observed_counts,
                        _lambda=optimizer_lambda,
                        )

                optimized_counts, optimized_objective = \
                    optimize_counts_optimized(
                        target_distribution=target_distribution,
                        observed_counts=observed_counts,
                        _lambda=optimizer_lambda,
                    )

                assert_feasible_allocation(
                    self,
                    standard_counts,
                    observed_counts,
                )

                assert_feasible_allocation(
                    self,
                    optimized_counts,
                    observed_counts,
                )

                self.assertAlmostEqual(
                    standard_objective,
                    optimized_objective,
                    delta=OBJECTIVE_TOLERANCE,
                    msg=(
                        f"Different objective values in scenario "
                        f"{scenario_id}.\n"
                        f"Target: {target_distribution}\n"
                        f"Available: {observed_counts}\n"
                        f"Lambda: {optimizer_lambda}\n"
                        f"Standard counts: {standard_counts}\n"
                        f"Optimized counts: {optimized_counts}\n"
                        f"Standard objective: {standard_objective}\n"
                        f"Optimized objective: {optimized_objective}"
                    ),
                )

                if standard_counts == optimized_counts:
                    exact_count_matches += 1
                else:
                    tied_count_differences += 1

        print(
            "\nRandom optimizer comparison:"
            f"\n  scenarios: {NUMBER_OF_RANDOM_SCENARIOS}"
            f"\n  exact count matches: {exact_count_matches}"
            f"\n  different counts with equal scores: "
            f"{tied_count_differences}"
        )

    def test_selected_edge_cases(self):
        scenarios = [
            {
                "target": {
                    "A": 0.5,
                    "B": 0.5,
                },
                "available": {
                    "A": 10,
                    "B": 10,
                },
                "lambda": 0.0,
            },
            {
                "target": {
                    "A": 0.5,
                    "B": 0.5,
                },
                "available": {
                    "A": 10,
                    "B": 10,
                },
                "lambda": 1.0,
            },
            {
                "target": {
                    "A": 0.9,
                    "B": 0.1,
                },
                "available": {
                    "A": 1,
                    "B": 20,
                },
                "lambda": 0.5,
            },
            {
                "target": {
                    "A": 0.01,
                    "B": 0.09,
                    "C": 0.90,
                },
                "available": {
                    "A": 20,
                    "B": 20,
                    "C": 1,
                },
                "lambda": 0.95,
            },
            {
                "target": {
                    "A": 1.0 / 3.0,
                    "B": 1.0 / 3.0,
                    "C": 1.0 / 3.0,
                },
                "available": {
                    "A": 1,
                    "B": 1,
                    "C": 1,
                },
                "lambda": 0.5,
            },
        ]

        for scenario_id, scenario in enumerate(scenarios):
            with self.subTest(scenario=scenario_id):
                standard_counts, standard_objective = \
                    optimize_counts(
                        target_distribution=scenario["target"],
                        observed_counts=scenario["available"],
                        _lambda=scenario["lambda"],
                    )

                optimized_counts, optimized_objective = \
                    optimize_counts_optimized(
                        target_distribution=scenario["target"],
                        observed_counts=scenario["available"],
                        _lambda=scenario["lambda"],
                    )
                
                self.assertAlmostEqual(
                    standard_objective,
                    optimized_objective,
                    delta=OBJECTIVE_TOLERANCE,
                )

