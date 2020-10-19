from collections import Counter

import numpy as np

from pastml.annotation import get_forest_stats
from pastml.ml import get_pij_method
from pastml import MODEL_ID


def simulate_states(tree, model, frequencies, kappa, tau, sf, character, rate_matrix=None, n_repetitions=1_000,
                    root_state_id=None):
    n_states = len(frequencies[0])
    state_ids = np.array(range(n_states))

    avg_br_len, num_nodes, num_tips, tree_len = get_forest_stats([tree])
    num_edges = num_nodes - 1
    tau_factor = tree_len / (tree_len + tau * num_edges)
    get_pij = get_pij_method(model, frequencies, kappa, rate_matrix=rate_matrix)

    for n in tree.traverse('levelorder'):
        model_id = getattr(n, MODEL_ID, 0)
        if n.is_root():
            if root_state_id is None:
                random_states = np.random.choice(state_ids, size=n_repetitions, p=frequencies[model_id])
            else:
                random_states = np.array([root_state_id] * n_repetitions)
        else:
            probs = get_pij[model_id]((n.dist + tau) * tau_factor * sf[model_id])
            probs = np.maximum(probs, 0)
            random_states = np.zeros(n_repetitions, dtype=int)
            parent_states = getattr(n.up, character)
            sorted_indices = np.argsort(parent_states)
            parent_state_nums = Counter(parent_states)
            offset = 0
            for i in state_ids:
                parent_nums_i = parent_state_nums[i]
                if parent_nums_i > 0:
                    random_states[sorted_indices[offset: offset + parent_nums_i]] = \
                        np.random.choice(state_ids, size=parent_nums_i, p=probs[i])
                    offset += parent_nums_i
        n.add_feature(character, random_states)

    return tree




