import unittest

import numpy as np
from ete3 import Tree

from pastml import get_personalized_feature_name
from pastml.ml import ALLOWED_STATES, JOINT_STATE, LH, choose_ancestral_states_mppa


class MPPAForceJointTest(unittest.TestCase):

    def test_force_joint_includes_joint_state(self):
        states = np.array(['A', 'B', 'C'])
        feature = 'state'
        lh_feature = get_personalized_feature_name(feature, LH)
        joint_feature = get_personalized_feature_name(feature, JOINT_STATE)
        allowed_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

        # Marginal probabilities [0.85, 0.10, 0.05] with joint state at index 2 ('C').
        # With force_joint=True, Brier score is minimized at k=2 for {'A', 'C'}.
        tree_forced = Tree('(a:1,b:1)r;', format=1)
        for node in tree_forced.traverse():
            node.add_feature(lh_feature, np.array([0.85, 0.10, 0.05]))
            node.add_feature(joint_feature, 2)

        choose_ancestral_states_mppa(tree_forced, feature, states, force_joint=True)
        root_allowed = getattr(tree_forced, allowed_feature)
        self.assertSetEqual(set(states[root_allowed == 1]), {'A', 'C'})

        # With force_joint=False, Brier score is minimized at k=1 for {'A'}.
        tree_unforced = Tree('(a:1,b:1)r;', format=1)
        for node in tree_unforced.traverse():
            node.add_feature(lh_feature, np.array([0.85, 0.10, 0.05]))
            node.add_feature(joint_feature, 2)

        choose_ancestral_states_mppa(tree_unforced, feature, states, force_joint=False)
        root_allowed_unforced = getattr(tree_unforced, allowed_feature)
        self.assertSetEqual(set(states[root_allowed_unforced == 1]), {'A'})

    def test_force_joint_includes_joint_state_at_index_zero(self):
        states = np.array(['A', 'B', 'C'])
        feature = 'state'
        lh_feature = get_personalized_feature_name(feature, LH)
        joint_feature = get_personalized_feature_name(feature, JOINT_STATE)
        allowed_feature = get_personalized_feature_name(feature, ALLOWED_STATES)

        tree = Tree('(a:1,b:1)r;', format=1)
        for node in tree.traverse():
            node.add_feature(lh_feature, np.array([0.05, 0.85, 0.10]))
            node.add_feature(joint_feature, 0)

        choose_ancestral_states_mppa(tree, feature, states, force_joint=True)
        root_allowed = getattr(tree, allowed_feature)
        self.assertSetEqual(set(states[root_allowed == 1]), {'A', 'B'})
