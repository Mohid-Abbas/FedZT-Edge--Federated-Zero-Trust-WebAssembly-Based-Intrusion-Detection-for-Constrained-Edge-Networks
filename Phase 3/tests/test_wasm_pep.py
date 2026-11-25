import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.layer1_wasm_pep import WasmPolicyEnforcementPoint
from src.layer2_federated import EdgeIDSModel

class TestWasmPEP(unittest.TestCase):
    def test_fuel_metering_and_inference(self):
        pep = WasmPolicyEnforcementPoint(client_id=1, execution_mode="wasm")
        model = EdgeIDSModel(input_dim=10)
        X_sample = np.random.randn(5, 10).astype(np.float32)
        
        res = pep.run_inference(model, X_sample)
        self.assertEqual(len(res["anomaly_probs"]), 5)
        self.assertGreater(res["fuel_consumed"], 0)
        self.assertEqual(res["mode"], "wasm")
        self.assertGreater(pep.total_fuel_consumed, 0)
        self.assertEqual(pep.inference_count, 5)

    def test_native_vs_wasm_mode(self):
        pep_wasm = WasmPolicyEnforcementPoint(client_id=1, execution_mode="wasm")
        pep_nat = WasmPolicyEnforcementPoint(client_id=1, execution_mode="native")
        model = EdgeIDSModel(input_dim=10)
        X = np.random.randn(10, 10).astype(np.float32)
        
        r_wasm = pep_wasm.run_inference(model, X)
        r_nat = pep_nat.run_inference(model, X)
        
        # Wasm has fuel metering, native does not
        self.assertGreater(r_wasm["fuel_consumed"], 0)
        self.assertEqual(r_nat["fuel_consumed"], 0)
        # Wasm binary size is smaller than native
        self.assertLess(r_wasm["binary_size_kb"], r_nat["binary_size_kb"])

    def test_local_fallback_snapshot(self):
        pep = WasmPolicyEnforcementPoint(client_id=2)
        # Critical attack
        action_attack = pep.evaluate_local_fallback(0.95)
        self.assertEqual(action_attack, 3) # Quarantine
        
        # Moderate anomaly
        action_suspicious = pep.evaluate_local_fallback(0.80)
        self.assertEqual(action_suspicious, 2) # Throttle
        
        # Benign
        action_benign = pep.evaluate_local_fallback(0.20)
        self.assertEqual(action_benign, 0) # Allow
        self.assertEqual(pep.fallback_invocations, 3)

if __name__ == "__main__":
    unittest.main()
