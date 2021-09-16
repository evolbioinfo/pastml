from collections import Counter

import numpy as np

from pastml import MODEL_ID, SKYLINE
from pastml.annotation import get_forest_stats
from pastml.ml import get_pij_kwargs, _get_p_ij_child


def simulate_states(tree, model, frequencies, kappa, tau, sf, character, rate_matrix=None, n_repetitions=1_000,
                    root_state_id=None, skyline_mapping=None):
    state_ids = [np.array(range(len(_))) for _ in frequencies]

    avg_br_len, num_nodes, num_tips, tree_len = get_forest_stats([tree])
    num_edges = num_nodes - 1
    tau_factor = tree_len / (tree_len + tau * num_edges)
    pij_kwargs = [get_pij_kwargs(model, frequencies=freq, kappa=k, rate_matrix=rate_matrix)
                  for freq, k in zip(frequencies, kappa if kappa is not None else [None] * len(frequencies))]

    for n in tree.traverse('levelorder'):
        if getattr(n, SKYLINE, False):
            continue
        model_id = getattr(n, MODEL_ID, 0)
        if n.is_root():
            if root_state_id is None:
                random_states = np.random.choice(state_ids[model_id], size=n_repetitions, p=frequencies[model_id])
            else:
                random_states = np.array([root_state_id] * n_repetitions)
            n.add_feature(character, random_states)

        non_skyline_children = []
        parent_states = getattr(n, character)
        sorted_indices = np.argsort(parent_states)
        parent_state_nums = Counter(parent_states)
        model_id = getattr(n, MODEL_ID, 0)
        for child in n.children:
            p_ij, non_skyline_child = _get_p_ij_child(child, tau, tau_factor, sf, model, pij_kwargs,
                                                      skyline_mapping)
            non_skyline_children.append(non_skyline_child)
            probs = np.maximum(p_ij, 0)
            random_states = np.zeros(n_repetitions, dtype=int)
            offset = 0
            for i in state_ids[model_id]:
                parent_nums_i = parent_state_nums[i]
                if parent_nums_i > 0:
                    random_states[sorted_indices[offset: offset + parent_nums_i]] = \
                        np.random.choice(state_ids[getattr(non_skyline_child, MODEL_ID, 0)],
                                         size=parent_nums_i, p=probs[i])
                    offset += parent_nums_i
            non_skyline_child.add_feature(character, random_states)

    return tree
