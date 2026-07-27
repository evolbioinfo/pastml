import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import time

GEOMETRIC = 'geometric'
ARITHMETIC = 'arithmetic'
ALPHA = 0.5
_LAMBDA = 0.5


def sampling_bias_correction_pipeline(
    df: pd.DataFrame,
    freq_priors: str | Path,
    data_sep: str,
    columns: list[str],
    alpha: float = ALPHA,
    interpolation_method: str = GEOMETRIC,
    _lambda: float = _LAMBDA,
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
        Path to a delimited file containing exactly two columns: the geographic
        state column and a prior-frequency column.

    data_sep
        Column separator used in the prior-frequency file.

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

    optimizer_lambda
        Trade-off parameter between matching the target distribution and
        retaining sequences. Must lie in the interval [0, 1].

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
    logger.debug('Sampling bias correction option enabled. Optimizing subsampling frequences ...')

    begin_time = time.time()

    assert len(freq_priors) > 0
    freq_priors_data = pd.read_csv(freq_priors, sep=data_sep, header=0)
    freq_priors_data_cols = freq_priors_data.columns
    
    if not columns[0] in freq_priors_data_cols:
        # TODO: complete the description
        raise ValueError(f"Column '{columns[0]}' does not appear in '{freq_priors}' file. "
                            "This file should have two columns ... (complete) ...")

    if not len(freq_priors_data_cols) == 2:
        raise ValueError(f"File '{freq_priors}' should contain exactly two columns, "
                            f"one of them named as '{columns[0]}'")

    freq_column = next(col for col in freq_priors_data.columns if col != columns[0])
    
    # Check: only numeric non-missing data
    if not is_numeric_dtype(freq_priors_data[freq_column]) \
        or freq_priors_data[freq_column].isna().any():
        
        raise ValueError(
            f"Column '{freq_column}' in '{freq_priors}' must contain only numeric, non-missing values."
        )
    
    # Check: frequencies must sum to 1.0
    freq_sum = freq_priors_data[freq_column].sum()
    if not np.isclose(freq_sum, 1.0):
        raise ValueError(
            f"Values in column '{freq_column}' in '{freq_priors}' sum to "
            f"{freq_sum}, but they must sum to 1.0."
        )
    
    # prior and observed distribitons into dictionaries
    freq_priors_dict = freq_priors_data.set_index(columns[0])[freq_column].to_dict()
    obs_dict = df[columns[0]].value_counts(normalize=True).to_dict()
    obs_counts_dict = df[columns[0]].value_counts(normalize=False).to_dict()

    target_dist_dict = interpolate_distributions(observed_distribution = obs_dict, 
                                                    prior_distribution = freq_priors_dict, 
                                                    alpha = alpha, 
                                                    method = interpolation_method)

    optimized_counts = optimize_counts(target_distribution = target_dist_dict,
                                        observed_counts = obs_counts_dict,
                                        _lambda = _lambda)

    #print(optimized_counts)

    m_opt = sum(optimized_counts.values())

    end_time = time.time()
    elapsed_time = round(end_time - begin_time,2)
    logger.debug(f"Completed. Best M = {m_opt} Elapsed time {elapsed_time} s")

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

    if method not in {ARITHMETIC, GEOMETRIC}:
        raise ValueError(
            f"method must be either '{ARITHMETIC}' or '{GEOMETRIC}', "
            f"but received '{method}'."
        )

    observed_labels = set(observed_distribution)
    prior_labels = set(prior_distribution)

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


# TODO: add a M_min and M_max parameters
def optimize_counts(
    target_distribution: dict[str, float],
    observed_counts: dict[str, int],
    _lambda: float,
) -> dict[str, int]:
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

        print(f"M = {final_size}, objective = {objective}")

        if objective < best_objective:
            best_objective = objective
            best_counts = counts.copy()

    return {
        label: int(count)
        for label, count in zip(labels, best_counts)
    }


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

from enum import Enum

class PruneMethod(str, Enum):
    RETAIN_LONGEST_TIP_BRANCHES = "RETAIN_LONGEST_TIP_BRANCHES"
    RETAIN_SHORTEST_TIP_BRANCHES = "RETAIN_SHORTEST_TIP_BRANCHES"
    RETAIN_LONGEST_DIST_TO_ROOT = "RETAIN_LONGEST_DIST_TO_ROOT"
    RETAIN_SHORTEST_DIST_TO_ROOT = "RETAIN_SHORTEST_DIST_TO_ROOT"

def prune_tree_by_state_counts(
    root,
    optimized_counts: dict[str, int],
    state_column: str,
    copy_tree: bool = False,
    method: PruneMethod = PruneMethod.RETAIN_SHORTEST_TIP_BRANCHES,
):
    """
    Prune a phylogenetic tree to retain specified leaf counts per state.

    Pruning options/methods:
    - ``RETAIN_SHORTEST_TIP_BRANCHES``:
      retain leaves with the shortest terminal branches.
    - ``RETAIN_LONGEST_TIP_BRANCHES``:
      retain leaves with the longest terminal branches.
    - ``RETAIN_SHORTEST_DIST_TO_ROOT``:
      retain leaves with the smallest root-to-tip distances.
    - ``RETAIN_LONGEST_DIST_TO_ROOT``:
      retain leaves with the largest root-to-tip distances.

    After selecting the requested number of leaves per state, all remaining
    leaves are removed. Suppressible internal nodes are removed while
    preserving branch lengths.

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
        Criterion used to rank leaves within each state. Must be one of the
        values in ``PRUNE_METHODS``.

    Returns
    -------
    ETE TreeNode
        The pruned tree.
    """
    if not optimized_counts:
        raise ValueError("optimized_counts must not be empty.")

    if not state_column:
        raise ValueError("state_column must be a non-empty string.")

    tree = root.copy(method="deepcopy") if copy_tree else root

    leaves_by_state: dict[str, list] = {}

    for leaf in tree.iter_leaves():

        # TODO: see how to deal with empty leaves
        if not hasattr(leaf, state_column):
            continue

        state = getattr(leaf, state_column)

        if state is None or state == "":
            continue

        # PastML state annotations are generally stored as sets.
        # TODO: see how to deal with leafs of ambiguous states
        # Probably I will have to go to the function where I calculate the empirical distribution
        # and exclude ambiguous levaes from the computation
        if isinstance(state, set):
            if len(state) != 1:
                continue

            state = next(iter(state))

        leaves_by_state.setdefault(state, []).append(leaf)
    
    # Validate the requested counts.
    for state, requested_count in optimized_counts.items():
        
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
            "This is because, probably, no prior frequencies were specified for the above states as well. "
            "Please check the prior frequencies file"
        )

    unknown_states = optimized_states - tree_states

    if unknown_states:
        raise ValueError(
            "The following states occur in optimized_counts but not among "
            f"the tree leaves: {sorted(unknown_states)}."
        )

    if method in {
        PruneMethod.RETAIN_LONGEST_TIP_BRANCHES,
        PruneMethod.RETAIN_SHORTEST_TIP_BRANCHES,
    }:
        distance_function = lambda leaf: float(leaf.dist)
    else:
        distance_function = lambda leaf: float(leaf.get_distance(tree))

    retain_longest = method in {
        PruneMethod.RETAIN_LONGEST_TIP_BRANCHES,
        PruneMethod.RETAIN_LONGEST_DIST_TO_ROOT
    }

    leaves_to_keep = []

    for state, leaves in leaves_by_state.items():
        requested_count = optimized_counts[state]

        if requested_count == 0:
            continue

        sorted_leaves = sorted(
            leaves,
            key=lambda leaf: (
                distance_function(leaf),
                str(leaf.name)
            ),
            reverse=retain_longest
        )

        leaves_to_keep.extend(sorted_leaves[:requested_count])

    if not leaves_to_keep:
        raise ValueError(
            "The optimized counts would remove every leaf from the tree."
        )

    tree.prune(
        leaves_to_keep,
        preserve_branch_length=True,
    )

    return tree
