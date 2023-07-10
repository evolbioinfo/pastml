from collections import Counter

import numpy as np


def simulate_states(tree, model, character, n_repetitions=1_000):
    n_states = len(model.states)
    state_ids = np.array(range(n_states))

    get_pij = model.get_Pij_t

    for n in tree.traverse('levelorder'):
        if n.is_root():
            random_states = np.random.choice(state_ids, size=n_repetitions, p=model.frequencies)
        else:
            probs = get_pij(n.dist)
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
