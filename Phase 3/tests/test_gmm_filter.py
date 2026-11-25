import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.layer2_federated import FederatedServer, get_parameters_vector

class TestGMMFilterAndFL(unittest.TestCase):
    def test_gmm_filter_clean_updates(self):
        server = FederatedServer(input_dim=10, use_defense=True)
        # Create 8 benign updates with small deltas around 0
        updates = []
        for i in range(8):
            delta = np.random.normal(0, 0.05, 100)
            weights = np.ones(100) + delta
            updates.append({
                "client_id": i,
                "weights": weights,
                "delta": delta,
                "num_samples": 100,
                "is_malicious": False
            })
            
        accepted, metrics = server.filter_updates_gmm_kmeans(updates)
        self.assertGreaterEqual(metrics["benign_preservation"], 0.70)
        self.assertGreaterEqual(len(accepted), 5)

    def test_gmm_filter_detects_poisoning(self):
        server = FederatedServer(input_dim=10, use_defense=True)
        updates = []
        # 7 Benign updates
        for i in range(7):
            delta = np.random.normal(0, 0.02, 100)
            updates.append({
                "client_id": i,
                "weights": np.ones(100) + delta,
                "delta": delta,
                "num_samples": 100,
                "is_malicious": False
            })
        # 3 Poisoned updates with large inverted deltas
        for i in range(7, 10):
            delta = np.random.normal(-5.0, 0.5, 100)
            updates.append({
                "client_id": i,
                "weights": np.ones(100) + delta,
                "delta": delta,
                "num_samples": 100,
                "is_malicious": True
            })
            
        accepted, metrics = server.filter_updates_gmm_kmeans(updates)
        # Verify malicious updates were flagged
        accepted_ids = [u["client_id"] for u in accepted]
        self.assertNotIn(7, accepted_ids)
        self.assertNotIn(8, accepted_ids)
        self.assertGreater(metrics["filter_recall"], 0.5)

    def test_fedavg_aggregation_math(self):
        server = FederatedServer(input_dim=5, aggregation_method="FedAvg", use_defense=False)
        base_w = get_parameters_vector(server.global_model)
        w1 = base_w + 1.0
        w2 = base_w + 3.0
        updates = [
            {"client_id": 0, "weights": w1, "delta": w1 - base_w, "num_samples": 100, "is_malicious": False},
            {"client_id": 1, "weights": w2, "delta": w2 - base_w, "num_samples": 100, "is_malicious": False}
        ]
        # Equal weights average: base_w + 2.0
        expected = base_w + 2.0
        new_w, _ = server.aggregate(updates)
        np.testing.assert_allclose(new_w, expected, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
