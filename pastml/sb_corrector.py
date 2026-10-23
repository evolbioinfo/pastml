import logging
import json
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import time
from enum import Enum
import heapq
from concurrent.futures import ThreadPoolExecutor


class PruneMethod(str, Enum):
    PRUNE_LONGEST_TIP_BRANCHES = "PRUNE_LONGEST_TIP_BRANCHES"
    PRUNE_SHORTEST_TIP_BRANCHES = "PRUNE_SHORTEST_TIP_BRANCHES"
    PRUNE_LONGEST_DIST_TO_ROOT = "PRUNE_LONGEST_DIST_TO_ROOT"
    PRUNE_SHORTEST_DIST_TO_ROOT = "PRUNE_SHORTEST_DIST_TO_ROOT"

GEOMETRIC = 'geometric'
ARITHMETIC = 'arithmetic'
ALPHA = 0.5
_LAMBDA = 0.5

def read_frequency_priors(
    freq_priors: str | Path,
    freq_priors_format: str,
) -> dict[str, dict[str, float]]:
    """
    Read prior state frequencies from JSON, CSV, or TSV.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from annotation columns to state-frequency dictionaries.
    """

    fmt = freq_priors_format.upper()

    if fmt == 'JSON':
        with open(freq_priors, 'r', encoding='utf-8') as f:
            priors = json.load(f)

        # Validate the common dictionary representation.
        if not isinstance(priors, dict):
            raise ValueError(
                "Frequency priors must map annotation columns "
                "to state-frequency dictionaries."
            )

    elif fmt in ('CSV', 'TSV'):
        sep = ',' if fmt == 'CSV' else '\t'

        data = pd.read_csv(
            freq_priors,
            sep=sep,
            dtype={'Column': str, 'State': str},
            keep_default_na=False,
        )

        expected_columns = {'Column', 'State', 'Frequency'}

        if set(data.columns) != expected_columns:
            raise ValueError(
                f"File '{freq_priors}' must contain exactly three "
                "columns: 'Column', 'State', and 'Frequency'."
            )

        if (data[['Column', 'State']] == '').any().any():
            raise ValueError(
                f"File '{freq_priors}' contains missing column or state names."
            )

        if data.duplicated(subset=['Column', 'State']).any():
            raise ValueError(
                f"File '{freq_priors}' contains duplicate "
                "column-state combinations."
            )

        priors = {
            column: dict(zip(group['State'], group['Frequency']))
            for column, group in data.groupby('Column', sort=False)
        }

    else:
        raise ValueError(
            f"Unsupported frequency priors format: '{fmt}'. "
            "Expected JSON, CSV, or TSV."
        )

    validated_priors = {}
    
    for column, distribution in priors.items():
        
        if not isinstance(column, str) or not column:
            raise ValueError("Invalid annotation column name.")

        if not isinstance(distribution, dict) or not distribution:
            raise ValueError(
                f"Prior distribution for '{column}' must "
                "be a non-empty dictionary."
            )

        validated_distribution = {}

        for state, frequency in distribution.items():

            if not isinstance(state, str) or not state:
                raise ValueError(
                    f"Invalid state name in column '{column}'."
                )

            try:
                if isinstance(frequency, (bool, dict, list)):
                    raise TypeError
                value = float(frequency)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid frequency for state '{state}' "
                    f"in column '{column}': {frequency!r}."
                )

            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"Frequency for state '{state}' in "
                    f"column '{column}' must be between 0 and 1."
                )

            validated_distribution[state] = value

        freq_sum = sum(validated_distribution.values())

        if not np.isclose(freq_sum, 1.0):
            raise ValueError(
                f"Prior frequencies for column '{column}' "
                f"sum to {freq_sum}, but must sum to 1.0."
            )

        validated_priors[column] = validated_distribution

    return validated_priors

def optimize_column_counts(
    column: str,
    df: pd.DataFrame,
    prior_distribution: dict[str, float],
    alpha: float,
    interpolation_method: str,
    _lambda: float,
) -> tuple[str, dict[str, int] | None]:
    """
    Optimize the retained sequence counts for one annotation column.
    """

    logger = logging.getLogger('pastml')

    # Calculate the observed frequencies and counts.
    obs_dict = df[column].value_counts(normalize=True).to_dict()

    obs_counts_dict = df[column].value_counts(
        normalize=False
    ).to_dict()

    # Construct the target distribution.
    target_dist_dict = interpolate_distributions(
        observed_distribution=obs_dict,
        prior_distribution=prior_distribution,
        alpha=alpha,
        method=interpolation_method,
    )

    # Optimize sequence counts.
    optimized_counts, _ = optimize_counts_optimized(
        target_distribution=target_dist_dict,
        observed_counts=obs_counts_dict,
        _lambda=_lambda,
    )

    m_opt = sum(optimized_counts.values())
    m_obs = sum(obs_counts_dict.values())

    logger.debug(
        f"Column '{column}': M_optimal = {m_opt}, "
        f"M_observed = {m_obs}"
    )

    # No pruning is required.
    if m_opt == m_obs:
        return column, None

    return column, optimized_counts


def sampling_bias_correction_pipeline(
    df: pd.DataFrame,
    freq_priors: str | Path,
    freq_priors_format: str,
    columns: list[str],
    alpha: float = ALPHA,
    interpolation_method: str = GEOMETRIC,
    _lambda: float = _LAMBDA,
    threads: int = 1,
) -> dict[str, int]:
    """
    Calculate sampling-bias-corrected sequence counts for each geographic state.

    The function reads a prior geographic-frequency distribution from a file,
    calculates the observed distribution from the input metadata, interpolates
    between the observed and prior distributions, and optimizes the number of
    sequences retained for each state.

    Parameters
    ----------
    df
        Metadata table containing the geographic-state column.

    freq_priors
        Path to a prior-frequencies column.

    freq_priors_format
        JSON, CSV, or TSV (str)

    columns
        Sequence containing the metadata columns used for ancestral-state
        reconstruction. The first element identifies the geographic-state
        column used for sampling-bias correction.

    alpha
        Interpolation strength between the observed and prior distributions.
        Must lie in the interval [0, 1].

    interpolation_method
        Interpolation method used to construct the target distribution, such
        as ``"arithmetic"`` or ``"geometric"``.

    _lambda
        Trade-off parameter between matching the target distribution and
        retaining sequences. Must lie in the interval [0, 1].

    threads
        Number of threads to be used (Default: 1)
        Parallelization only takes place across independent columns, hence
        if a single column is specified, only one thread is used.
    
    Returns
    -------
    dict[str, int]
        Optimized number of sequences to retain for each geographic state.

    Raises
    ------
    ValueError
        If the input columns, prior frequencies, distributions, or parameter
        values are invalid.
    """

    logger = logging.getLogger('pastml')
    logger.debug('Sampling bias correction option enabled. Optimizing subsampling frequences.')
    logger.debug('Alpha value {}, lambda = {}, interpolation {}'.format(alpha, _lambda, interpolation_method))

    begin_time = time.time()

    # Read the frequency priors once.
    freq_priors_dict = read_frequency_priors(
        freq_priors=freq_priors,
        freq_priors_format=freq_priors_format,
    )

    # Ensure all requested columns exist in the metadata.
    missing_columns = set(columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"Annotation columns not found in the metadata: "
            f"{sorted(missing_columns)}"
        )

    # Priors for unselected columns are ignored.
    unknown_columns = set(freq_priors_dict) - set(columns)

    if unknown_columns:
        logger.warning(
            f"Prior frequencies specified for unselected "
            f"columns {sorted(unknown_columns)}: they will be ignored. "
            f"The specified columns are: {columns}."
        )

    # Only optimize columns with explicitly specified priors.
    columns_with_priors = [
        column for column in dict.fromkeys(columns)
        if column in freq_priors_dict
    ]

    if not columns_with_priors:
        logger.debug(
            "No prior frequencies specified for selected columns."
        )
        return {}

    # Never create more workers than independent tasks.
    threads = min(threads, len(columns_with_priors))

    logger.debug(
        f"Optimizing {len(columns_with_priors)} columns "
        f"using {threads} thread(s)."
    )

    optimized_counts = {}

    def worker(column):
        return optimize_column_counts(
            column=column,
            df=df,
            prior_distribution=freq_priors_dict[column],
            alpha=alpha,
            interpolation_method=interpolation_method,
            _lambda=_lambda,
        )

    with ThreadPoolExecutor(max_workers=threads) as executor:

        # executor.map preserves input ordering.
        results = executor.map(worker, columns_with_priors)
        
        for column, counts in results:
            if counts is not None:
                optimized_counts[column] = counts

    elapsed_time = round(time.time() - begin_time, 2)

    logger.debug(
        f"Computing optimized frequencies completed in "
        f"{elapsed_time} s."
    )
    
    return optimized_counts



# TODO: Optimize
# I have to optimize this function.
# IDEAS:
# - Basically, this is a curved function, once it finds the optimum, it start increasing. I should stop there and not test further values.
def interpolate_distributions(
    observed_distribution: dict[str, float],
    prior_distribution: dict[str, float],
    alpha: float,
    method: str,
) -> dict[str, float]:
    """
    Interpolate between the observed and the prior probability distribution.

    Parameters
    ----------
    observed_distribution
        Observed probabilities, indexed by label.

    prior_distribution
        Prior probabilities, indexed by label. The labels must be identical
        to those in observed_distribution.

    alpha
        Interpolation strength in [0, 1].
        alpha = 0 returns the observed distribution.
        alpha = 1 returns the prior distribution.

    method
        Interpolation method: "arithmetic" or "geometric
    
    Returns
    -------
    dict[str, float]
        The normalized target distribution.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"alpha parameter (--alpha) must be between 0.0 and 1.0, but received {alpha}."
        )

    if method not in [ARITHMETIC, GEOMETRIC]:
        raise ValueError(
            f"method must be either '{ARITHMETIC}' or '{GEOMETRIC}', "
            f"but received '{method}'."
        )

    observed_distribution = {
        str(label): value
        for label, value in observed_distribution.items()
    }

    prior_distribution = {
        str(label): value
        for label, value in prior_distribution.items()
    }

    observed_labels = set(observed_distribution)
    prior_labels = set(prior_distribution)

    #print(observed_labels)
    #print(prior_labels)
    #print("HEY OP!")

    # TODO: for now we assume that observed and prior labels are identical
    # TODO: I'll see how I will handle non-identical sets later
    if observed_labels != prior_labels:
        raise ValueError(
            "The observed and prior distributions must contain exactly "
            "the same labels."
        )

    observed = {
        label: float(value)
        for label, value in observed_distribution.items()
    }

    prior = {
        label: float(value)
        for label, value in prior_distribution.items()
    }

    if any(not np.isfinite(value) or value < 0.0 for value in prior.values()):
        raise ValueError(
            "The prior distribution must contain only finite, "
            "non-negative values."
        )

    if method == ARITHMETIC:
        return {
            label: (
                (1.0 - alpha) * observed[label]
                + alpha * prior[label]
            )
            for label in observed
        }

    else:
        assert method == GEOMETRIC
    
    target = {
        label: (
            observed[label] ** (1.0 - alpha)
            * prior[label] ** alpha
        )
        for label in observed
    }

    normalization_constant = sum(target.values())

    # TODO: This is a case that I have to take into serious consideration
    # when assuming non-identical label sets
    if normalization_constant == 0.0:
        raise ValueError(
            "The geometric interpolation cannot be normalized. "
            "The observed and prior distributions have no label with "
            "positive probability in both distributions."
        )

    return {
        label: value / normalization_constant
        for label, value in target.items()
    }

def optimize_counts(
    target_distribution: dict[str, float],
    observed_counts: dict[str, int],
    _lambda: float,
) -> tuple[dict[str, int], float]:
    """
    Optimize integer state (e.g., country) counts 

    Parameters
    ----------
    target_distribution
        Target probability q_c for every state (e.g., country).

    observed_counts
        Number N_c of available sequences for every state (e.g., country).

    _lambda
        Trade-off parameter in [0, 1]. Larger values prioritize matching
        the target distribution, while smaller values prioritize retaining
        sequences.

    Returns
    -------
    dict[str, int]
        Optimized count m_c for every state (e.g., country).
    
    float
        The optimal criterion value 
    """
    
    # TODO: Implement also for the case where the is an non-empty intersection
    if set(target_distribution) != set(observed_counts):
        raise ValueError(
            "target_distribution and observed_counts must contain "
            "exactly the same country labels."
        )

    labels = list(target_distribution)

    target = np.asarray(
        [target_distribution[label] for label in labels],
        dtype=float,
    )

    available = np.asarray(
        [observed_counts[label] for label in labels],
        dtype=int,
    )

    if not np.isclose(target.sum(), 1.0):
        raise ValueError(
            f"target_distribution sums to {target.sum()}, not 1.0."
        )

    initial_size = int(available.sum())
    best_counts = None
    best_objective = np.inf

    for final_size in range(1, initial_size + 1):
        counts = bounded_integer_allocation(
            target=target,
            available=available,
            final_size=final_size,
        )

        achieved_distribution = counts / final_size

        geographic_error = np.mean(
            (achieved_distribution - target) ** 2
        )

        retention_error = (
            1.0 - final_size / initial_size
        ) ** 2

        objective = (
            _lambda * geographic_error
            + (1.0 - _lambda) * retention_error
        )

        #print(f"M = {final_size}, objective = {objective}")

        if objective < best_objective:
            best_objective = objective
            best_counts = counts.copy()

    return {
        label: int(count)
        for label, count in zip(labels, best_counts)
    }, best_objective

# TODO: add a M_min and M_max parameters
def optimize_counts_optimized(
    target_distribution: dict[str, float],
    observed_counts: dict[str, int],
    _lambda: float,
) -> tuple[dict[str, int], float]:
    """
    Optimize integer state (e.g., country) counts 

    Parameters
    ----------
    target_distribution
        Target probability q_c for every state (e.g., country).

    observed_counts
        Number N_c of available sequences for every state (e.g., country).

    _lambda
        Trade-off parameter in [0, 1]. Larger values prioritize matching
        the target distribution, while smaller values prioritize retaining
        sequences.

    Returns
    -------
    dict[str, int]
        Optimized count m_c for every state (e.g., country).
    
    float
        The optimal criterion value 
    """
    
    # TODO: Implement also for the case where the is an non-empty intersection
    if set(target_distribution) != set(observed_counts):
        raise ValueError(
            "target_distribution and observed_counts must contain "
            "exactly the same country labels."
        )

    labels = list(target_distribution)

    target = np.asarray(
        [target_distribution[label] for label in labels],
        dtype=float,
    )

    available = np.asarray(
        [observed_counts[label] for label in labels],
        dtype=int,
    )

    if not np.isclose(target.sum(), 1.0):
        raise ValueError(
            f"target_distribution sums to {target.sum()}, not 1.0."
        )

    initial_size = int(available.sum())
    best_counts = None
    best_objective = np.inf

    for final_size in range(initial_size, 0, -1):
        retention_error = (
            1.0 - final_size / initial_size
        ) ** 2

        # Safe lower bound: geographic error cannot be negative.
        objective_lower_bound = (
            (1.0 - _lambda)
            * retention_error
        )

        if objective_lower_bound >= best_objective:
            break

        counts = bounded_integer_allocation_optimized(
            target=target,
            available=available,
            final_size=final_size,
        )

        achieved_distribution = counts / final_size

        geographic_error = np.mean(
            (achieved_distribution - target) ** 2
        )

        objective = (
            _lambda * geographic_error
            + (1.0 - _lambda) * retention_error
        )

        #print(f"M = {final_size}, objective = {objective}")

        if objective < best_objective:
            best_objective = objective
            best_counts = counts.copy()

    if best_counts is None:
        raise RuntimeError(
            "No feasible count allocation was found."
        )

    return {
        label: int(count)
        for label, count in zip(labels, best_counts)
    }, best_objective


def bounded_integer_allocation(
    target: np.ndarray,
    available: np.ndarray,
    final_size: int,
) -> np.ndarray:
    """
    Allocate exactly final_size items while respecting upper bounds.

    At each step, assign one item to the country that yields the largest
    reduction in squared distance from its ideal quota.
    """

    if final_size > available.sum():
        raise ValueError(
            "final_size cannot exceed the total available count."
        )

    quotas = final_size * target
    counts = np.zeros(len(target), dtype=int)

    for _ in range(final_size):
        eligible = counts < available

        if not np.any(eligible):
            raise RuntimeError(
                "No country has remaining capacity."
            )

        current_error = (counts - quotas) ** 2
        updated_error = (counts + 1 - quotas) ** 2

        error_change = updated_error - current_error

        # Ineligible countries must never be selected.
        error_change[~eligible] = np.inf

        selected_country = int(np.argmin(error_change))
        counts[selected_country] += 1

    return counts


def bounded_integer_allocation_optimized(
    target: np.ndarray,
    available: np.ndarray,
    final_size: int,
) -> np.ndarray:
    """
    Allocate exactly ``final_size`` items while respecting state-specific
    upper bounds.

    The function minimizes

        sum_c (m_c - final_size * target_c) ** 2

    subject to

        0 <= m_c <= available_c
        sum_c m_c = final_size
        m_c integer.
    """
    total_available = int(available.sum())

    if not 0 <= final_size <= total_available:
        raise ValueError(
            f"final_size must be between 0 and {total_available}, "
            f"but received {final_size}."
        )

    quotas = final_size * target

    # Start near the unconstrained optimum rather than from zero.
    counts = np.minimum(
        np.floor(quotas).astype(int),
        available,
    )

    remaining = final_size - int(counts.sum())

    if remaining == 0:
        return counts

    # Each entry is:
    #     (marginal error increase, state index)
    heap = []

    for index in range(len(target)):
        if counts[index] < available[index]:
            marginal_cost = (
                2.0 * (counts[index] - quotas[index]) + 1.0
            )

            heapq.heappush(
                heap,
                (marginal_cost, index),
            )

    for _ in range(remaining):
        if not heap:
            raise RuntimeError(
                "No state has sufficient remaining capacity."
            )

        _, selected_index = heapq.heappop(heap)

        counts[selected_index] += 1

        # If this state still has capacity, insert its new marginal cost.
        if counts[selected_index] < available[selected_index]:
            new_marginal_cost = (
                2.0
                * (counts[selected_index] - quotas[selected_index])
                + 1.0
            )

            heapq.heappush(
                heap,
                (new_marginal_cost, selected_index),
            )

    return counts

# TODO: delete this function
def check_copy(root, copied_root, columns):

    nodes_orig = [n.name for n in root.traverse()]
    copied_nodes = [n.name for n in copied_root.traverse()]

    nodes_orig_id = [hex(id(node)) for node in root.traverse()]
    copied_nodes_id = [hex(id(node)) for node in copied_root.traverse()]

    label_name = columns[0]

    nodes_orig_labels = [getattr(n, label_name, "") if hasattr(n, columns[0]) else "" for n in root.traverse()]
    copied_nodes_labels = [getattr(n, label_name, "") if hasattr(n, columns[0]) else "" for n in copied_root.traverse()]
    
    is_leaf_nodes_orig = [True if n.is_leaf() else False for n in root.traverse()]
    is_leaf_nodes_copied = [True if n.is_leaf() else False for n in copied_root.traverse()]

    for i in range(len(nodes_orig)):
        assert nodes_orig[i] == copied_nodes[i]
        assert nodes_orig_id[i] != copied_nodes_id[i]
        assert nodes_orig_labels[i] == copied_nodes_labels[i]
        assert is_leaf_nodes_orig[i] == is_leaf_nodes_copied[i]

        if i < 50:
            print('{} ({}) - {} ({})'.format(nodes_orig[i], nodes_orig_id[i], copied_nodes[i], copied_nodes_id[i]))
            print('{} - {}'.format(nodes_orig_labels[i], copied_nodes_labels[i]))
            print('{} - {}'.format(is_leaf_nodes_orig[i], is_leaf_nodes_copied[i]))
    
    return

def prune_tree_by_state_counts(
    root,
    optimized_counts: dict[str, int],
    state_column: str,
    copy_tree: bool = False,
    method: PruneMethod = PruneMethod.PRUNE_LONGEST_TIP_BRANCHES,
):
    """
    Prune a phylogenetic tree to retain specified leaf counts per state.

    Pruning options/methods:

    - ``PRUNE_SHORTEST_TIP_BRANCHES``:
      iteratively prune the currently shortest eligible terminal branch.
      Terminal branch lengths are recomputed after every pruning operation.

    - ``PRUNE_LONGEST_TIP_BRANCHES``:
      iteratively prune the currently longest eligible terminal branch.
      Terminal branch lengths are recomputed after every pruning operation.

    - ``PRUNE_SHORTEST_DIST_TO_ROOT``:
      prune leaves having the shortest root-to-tip distances.

    - ``PRUNE_LONGEST_DIST_TO_ROOT``:
      prune leaves having the longest root-to-tip distances.

    A leaf is eligible for pruning only if removing it would leave at least
    ``optimized_counts[state]`` leaves for its state.

    Parameters
    ----------
    root
        Root node of an ETE phylogenetic tree.

    optimized_counts
        Number of leaves to retain for every state. For example:
        ``{"Greece": 20, "Italy": 15}``.

    state_column
        Name of the node feature containing the state annotation, such as
        ``"Country"`` or ``"subtype"``.

    copy_tree
        If True, prune a copy of the tree and leave the original unchanged.
        If False, prune the input tree in place.

    method
        Criterion used to choose leaves for pruning.

    Returns
    -------
    ETE TreeNode
        The pruned tree.
    """

    if not optimized_counts:
        raise ValueError("optimized_counts must not be empty.")

    if not state_column:
        raise ValueError("state_column must be a non-empty string.")

    try:
        method = PruneMethod(method)
    except ValueError:
        raise ValueError(
            f"Unknown pruning method '{method}'. "
            f"Expected one of: {[m.value for m in PruneMethod]}."
        )

    tree = root.copy(method="deepcopy") if copy_tree else root

    def get_leaf_state(leaf):
        """
        Return the single unambiguous state of a leaf.

        TODO:
        Decide how to handle leaves without annotations, empty states,
        and ambiguous states. For now, such leaves are ignored.
        """

        if not hasattr(leaf, state_column):
            return None

        state = getattr(leaf, state_column)

        if state is None or state == "":
            return None
        
        # PastML state annotations are generally stored as sets.
        #
        # TODO:
        # Decide how to handle ambiguous leaf states.
        # For now, leaves with multiple possible states are ignored.
        if isinstance(state, set):

            if len(state) != 1:
                return None

            state = next(iter(state))

        return state

    # ------------------------------------------------------------
    # Collect leaves by state.
    # ------------------------------------------------------------
    leaves_by_state: dict[str, list] = {}

    for leaf in tree.iter_leaves():

        state = get_leaf_state(leaf)

        if state is None:
            continue

        leaves_by_state.setdefault(state, []).append(leaf)

    # ------------------------------------------------------------
    # Validate requested counts.
    # ------------------------------------------------------------
    for state, requested_count in optimized_counts.items():

        if requested_count < 0:
            raise ValueError(
                f"Cannot retain a negative number of leaves for "
                f"state '{state}': {requested_count}."
            )

        available_count = len(leaves_by_state.get(state, []))

        if requested_count > available_count:
            raise ValueError(
                f"Cannot retain {requested_count} leaves for state '{state}': "
                f"only {available_count} are available. This may happen when "
                "too many leaves have missing or ambiguous states. Please "
                "check the annotation file."
            )

    tree_states = set(leaves_by_state)
    optimized_states = set(optimized_counts)

    missing_counts = tree_states - optimized_states

    if missing_counts:
        raise ValueError(
            "No optimized count was provided for the following states "
            f"present in the tree: {sorted(missing_counts)}. "
            "This is probably because no prior frequencies were specified "
            "for these states. Please check the prior frequencies file."
        )

    unknown_states = optimized_states - tree_states

    if unknown_states:
        raise ValueError(
            "The following states occur in optimized_counts but not among "
            f"the tree leaves: {sorted(unknown_states)}."
        )
    
    # ============================================================
    # Iterative terminal-branch pruning
    # ============================================================
    if method in {
        PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES,
        PruneMethod.PRUNE_LONGEST_TIP_BRANCHES,
    }:

        current_counts = {
            state: len(leaves)
            for state, leaves in leaves_by_state.items()
        }

        remaining_to_prune = sum(
            current_counts[state] - optimized_counts[state]
            for state in current_counts
        )

        prune_shortest = (
            method == PruneMethod.PRUNE_SHORTEST_TIP_BRANCHES
        )

        def leaf_priority(leaf):
            dist = float(leaf.dist)

            if prune_shortest:
                return dist

            return -dist

        def push_if_eligible(heap, leaf):
            state = get_leaf_state(leaf)

            if state is None:
                return

            if current_counts[state] <= optimized_counts[state]:
                return

            priority = leaf_priority(leaf)

            heapq.heappush(
                heap,
                (
                    priority,
                    str(leaf.name),
                    id(leaf),
                    leaf,
                    state,
                ),
            )

        heap = []

        for leaf in tree.iter_leaves():
            push_if_eligible(heap, leaf)

        while remaining_to_prune > 0:

            selected_leaf = None
            selected_state = None

            while heap:

                (
                    stored_priority,
                    _stored_name,
                    _stored_id,
                    candidate_leaf,
                    candidate_state,
                ) = heapq.heappop(heap)

                # Skip leaves already removed from the tree.
                if candidate_leaf.up is None:
                    continue

                current_state = get_leaf_state(candidate_leaf)

                if current_state is None:
                    continue

                if current_state != candidate_state:
                    continue

                if current_counts[current_state] <= optimized_counts[current_state]:
                    continue

                # Skip stale heap entries.
                #
                # This matters because when a sister leaf is pruned,
                # this leaf's terminal branch length may change. We then push
                # a fresh heap entry with the updated distance, but the old
                # entry may still exist in the heap.
                current_priority = leaf_priority(candidate_leaf)

                if current_priority != stored_priority:
                    continue

                selected_leaf = candidate_leaf
                selected_state = current_state
                break

            if selected_leaf is None:
                raise RuntimeError(
                    "No eligible leaf could be found although pruning "
                    "is still required."
                )

            parent = selected_leaf.up

            affected_sisters = []

            if parent is not None:
                affected_sisters = [
                    child
                    for child in parent.children
                    if child is not selected_leaf
                ]

            selected_leaf.delete(
                prevent_nondicotomic=True,
                preserve_branch_length=True,
            )

            current_counts[selected_state] -= 1
            remaining_to_prune -= 1

            # Only local neighbours can have changed terminal branch length.
            # Reinsert affected sister leaves with their updated distance.
            for sister in affected_sisters:

                if sister.up is None:
                    continue

                if not sister.is_leaf():
                    continue

                push_if_eligible(heap, sister)

        return tree

    # ============================================================
    # One-shot root-distance pruning
    # ============================================================

    prune_longest = (
        method == PruneMethod.PRUNE_LONGEST_DIST_TO_ROOT
    )

    leaves_to_prune = set()

    for state, leaves in leaves_by_state.items():

        requested_count = optimized_counts[state]
        available_count = len(leaves)

        n_to_prune = available_count - requested_count

        if n_to_prune == 0:
            continue

        sorted_leaves = sorted(
            leaves,
            key=lambda leaf: (
                float(leaf.get_distance(tree)),
                str(leaf.name),
            ),
            reverse=prune_longest,
        )

        leaves_to_prune.update(
            sorted_leaves[:n_to_prune]
        )

    leaves_to_keep = [
        leaf
        for leaf in tree.iter_leaves()
        if leaf not in leaves_to_prune
    ]

    if not leaves_to_keep:
        raise ValueError(
            "The optimized counts would remove every leaf from the tree."
        )

    tree.prune(
        leaves_to_keep,
        preserve_branch_length=True,
    )

    return tree

def get_inner_node_states(subsampled_forest, character):
    """
    Collect the states inferred on the inner nodes of the subsampled tree(s).

    Returns
    -------
    dict[str, set]
        Mapping from inner node names to their (possibly ambiguous) sets of states.
    """

    node_states = {}

    for root in subsampled_forest:
        for node in root.traverse():
            if node.is_leaf():
                continue

            states = getattr(node, character, set())

            if not states:
                continue

            # Defensive handling in case a single state is stored as a string.
            if isinstance(states, str):
                states = {states}

            node_states[node.name] = set(states)

    return node_states


def set_inner_node_states(forest, character, node_states):
    """
    Fix the states inferred on the subsampled tree(s) on the corresponding inner nodes of the original tree(s).
    Ambiguous states are kept as sets, i.e. they restrict the allowed states of the node.
    """

    logger = logging.getLogger('pastml')
    logger.debug("Mapping the states inferred on the subsampled tree(s) back to original tree(s)")

    for root in forest:
        for node in root.traverse():
            if node.is_leaf():
                continue

            if node.name in node_states:
                node.add_feature(character, set(node_states[node.name]))
