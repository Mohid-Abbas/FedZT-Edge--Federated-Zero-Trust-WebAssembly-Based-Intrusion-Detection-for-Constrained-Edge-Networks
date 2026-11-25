import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.layer3_drl_pdp import DRLTrustAgent, StaticThresholdPolicy

class TestDRLAgent(unittest.TestCase):
    def test_action_selection_and_costs(self):
        agent = DRLTrustAgent(state_dim=5, action_dim=4)
        state = np.array([0.9, 0.4, 0.5, 2.0, 0.7])
        action = agent.select_action(state, evaluate=True)
        self.assertIn(action, [0, 1, 2, 3])
        self.assertIn(agent.action_costs[action], [0.0, 0.25, 0.50, 1.00])

    def test_reward_mechanisms(self):
        agent = DRLTrustAgent(state_dim=5, action_dim=4)
        # Attack state properly quarantined (action=3)
        state_atk = np.array([0.95, 0.2, 0.6, 2.0, 0.5])
        reward_correct = agent.compute_reward(state_atk, action=3, true_label=1)
        # Attack missed (action=0)
        reward_breach = agent.compute_reward(state_atk, action=0, true_label=1)
        self.assertGreater(reward_correct, reward_breach)
        
        # Benign state allowed (action=0)
        state_benign = np.array([0.05, 0.9, 0.1, 0.0, 0.3])
        reward_benign_allowed = agent.compute_reward(state_benign, action=0, true_label=0)
        # Benign state falsely quarantined (action=3)
        reward_false_positive = agent.compute_reward(state_benign, action=3, true_label=0)
        self.assertGreater(reward_benign_allowed, reward_false_positive)

    def test_static_policy_ablation(self):
        static = StaticThresholdPolicy(allow_th=0.35, reauth_th=0.65, throttle_th=0.85)
        # Low anomaly -> Allow
        self.assertEqual(static.select_action([0.1]), 0)
        # Medium anomaly -> Reauth
        self.assertEqual(static.select_action([0.5]), 1)
        # High anomaly -> Throttle
        self.assertEqual(static.select_action([0.75]), 2)
        # Critical anomaly -> Quarantine
        self.assertEqual(static.select_action([0.95]), 3)

if __name__ == "__main__":
    unittest.main()
