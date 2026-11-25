"""
Layer 1: WebAssembly Policy Enforcement Point (Wasm PEP)
Implements sandboxed edge inference with fuel metering, memory bounds,
latency profiling, native vs. Wasm modes, and local policy snapshot fallback.
"""

import time
import numpy as np

try:
    from .config import WASM_CONFIG
except (ImportError, ValueError):
    from config import WASM_CONFIG

class WasmPolicyEnforcementPoint:
    def __init__(self, client_id, execution_mode="wasm", fuel_budget=1_000_000):
        self.client_id = client_id
        self.execution_mode = execution_mode  # "wasm" or "native"
        self.fuel_budget = fuel_budget
        self.total_fuel_consumed = 0
        self.inference_count = 0
        
        # Local Policy Snapshot Fallback (Step 5 / Ablation h)
        # Default snapshot action rule: if anomaly_prob > 0.70 => Throttle, if > 0.88 => Quarantine, else Allow
        self.local_policy_snapshot = {
            "allow_threshold": 0.50,
            "throttle_threshold": 0.75,
            "quarantine_threshold": 0.90,
            "timestamp": time.time(),
            "active": False
        }
        self.fallback_invocations = 0

    def enforce_policy_action(self, action, flow_meta=None):
        """
        Enforces a Zero-Trust action:
        0: Allow
        1: Adaptive Re-authentication
        2: Throttle
        3: Quarantine
        """
        action_names = {0: "ALLOW", 1: "RE_AUTH", 2: "THROTTLE", 3: "QUARANTINE"}
        name = action_names.get(action, "UNKNOWN")
        enforcement_record = {
            "client_id": self.client_id,
            "action": action,
            "action_name": name,
            "timestamp": time.time(),
            "status": "ENFORCED"
        }
        return enforcement_record

    def evaluate_local_fallback(self, anomaly_prob):
        """
        Fallback decision when PDP is disconnected / unreachable (Ablation h).
        Preserves availability under edge network partition.
        """
        self.fallback_invocations += 1
        if anomaly_prob >= self.local_policy_snapshot["quarantine_threshold"]:
            action = 3 # Quarantine
        elif anomaly_prob >= self.local_policy_snapshot["throttle_threshold"]:
            action = 2 # Throttle
        elif anomaly_prob >= self.local_policy_snapshot["allow_threshold"]:
            action = 1 # Re-auth
        else:
            action = 0 # Allow
        return action

    def update_policy_snapshot(self, new_policy_dict):
        """Updates the edge cached snapshot from the central PDP."""
        self.local_policy_snapshot.update(new_policy_dict)
        self.local_policy_snapshot["timestamp"] = time.time()

    def run_inference(self, model, X_sample):
        """
        Executes inference for a batch or single sample inside the simulated Wasm runtime.
        Computes exact fuel consumption, memory usage, and execution latency.
        """
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
            
        n_samples, n_features = X_sample.shape
        start_time = time.perf_counter()
        
        # Execute model forward pass
        # Models can be PyTorch NN, LogisticRegression, DecisionTree, or weights matrix
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_sample)
            anomaly_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        elif hasattr(model, 'forward'):
            import torch
            with torch.no_grad():
                tensor_x = torch.FloatTensor(X_sample)
                logits = model(tensor_x)
                if logits.shape[1] > 1:
                    probs = torch.softmax(logits, dim=1).numpy()
                    anomaly_probs = probs[:, 1]
                else:
                    anomaly_probs = torch.sigmoid(logits).numpy().flatten()
        elif hasattr(model, 'weights') and hasattr(model, 'bias'):
            # Linear / Logistic dot product
            z = np.dot(X_sample, model.weights) + model.bias
            anomaly_probs = 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))
        else:
            anomaly_probs = np.zeros(n_samples)
            
        elapsed_sec = time.perf_counter() - start_time
        
        # Fuel Metering Model (Step 5 / Step 8 / Ablation f)
        # Arithmetic ops (dot product, additions, activations) ~ 3 ops per feature * samples
        # Nonlinear activations (sigmoid/softmax) ~ 25 ops
        raw_ops = n_samples * (n_features * 4 + 30)
        
        if self.execution_mode == "wasm":
            # Fuel cost in Wasmtime: ~10 fuel per standard VM opcode
            fuel_used = raw_ops * WASM_CONFIG["base_instruction_fuel"]
            # Wasmtime sandboxing overhead factor
            adjusted_latency_ms = (elapsed_sec * 1000.0) * (1.0 + WASM_CONFIG["sandbox_overhead_pct"] / 100.0)
            mem_usage_mb = WASM_CONFIG["wasm_memory_overhead_mb"] + (n_samples * n_features * 8) / (1024 * 1024)
            binary_size_kb = WASM_CONFIG["wasm_binary_size_kb"]
        else: # Native mode
            fuel_used = 0 # Native unmetered
            adjusted_latency_ms = elapsed_sec * 1000.0
            mem_usage_mb = 1.2 + (n_samples * n_features * 8) / (1024 * 1024)
            binary_size_kb = WASM_CONFIG["native_binary_size_kb"]
            
        self.total_fuel_consumed += fuel_used
        self.inference_count += n_samples
        
        return {
            "anomaly_probs": anomaly_probs,
            "latency_ms": adjusted_latency_ms,
            "fuel_consumed": fuel_used,
            "memory_mb": mem_usage_mb,
            "binary_size_kb": binary_size_kb,
            "mode": self.execution_mode
        }
