import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.baselines import run_centralized_b1, run_local_wasm_only_b6, run_federated_pipeline

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic fast dataset for tests: linearly separable by feature 0
        # Use more samples and clear signal so small MLP can learn in 5 epochs
        n_train, n_test = 400, 100
        X_all = np.random.randn(n_train + n_test, 10).astype(np.float32)
        # Clear linear signal: class = 1 if feature_0 > 0
        y_all = (X_all[:, 0] > 0).astype(int)
        self.X_train = X_all[:n_train]
        self.y_train = y_all[:n_train]
        self.X_test  = X_all[n_train:]
        self.y_test  = y_all[n_train:]

    def test_b1_centralized(self):
        res = run_centralized_b1(self.X_train, self.y_train, self.X_test, self.y_test, epochs=5)
        self.assertEqual(res["baseline_id"], "B1")
        # MLP should achieve > 0.70 F1 on a simple 1-feature linear classification
        self.assertGreaterEqual(res["f1"], 0.70)

    def test_b6_local_wasm_only(self):
        res = run_local_wasm_only_b6(self.X_train, self.y_train, self.X_test, self.y_test, num_clients=3, epochs=2)
        self.assertEqual(res["baseline_id"], "B6")
        self.assertGreater(res["total_fuel"], 0)

    def test_fedzt_edge_treatment(self):
        res = run_federated_pipeline(
            self.X_train, self.y_train, self.X_test, self.y_test,
            config_id="Treatment_FedZT",
            num_clients=3,
            alpha=0.5,
            rounds=2,
            local_epochs=1,
            aggregation_method="FedProx",
            use_defense=True,
            use_drl=True,
            use_crypto=True,
            seed=42
        )
        self.assertEqual(res["baseline_id"], "Treatment_FedZT")
        self.assertGreaterEqual(res["f1"], 0.0)
        self.assertGreater(res["accuracy"], 0.0)
        self.assertGreater(res["mb_transferred"], 0.0)
        self.assertGreater(res["energy_joules_per_round"], 0.0)
        self.assertIsNotNone(res["drl_avg_reward"])

if __name__ == "__main__":
    unittest.main()
