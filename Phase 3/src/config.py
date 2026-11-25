"""
Central Configuration for FedZT-Edge (Phase 3)
Stores hyperparameters, paths, hardware specifications, and experimental grids.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
RAW_LOGS_DIR = os.path.join(RESULTS_DIR, "raw_logs")
REPORT_DIR = os.path.join(BASE_DIR, "report")

# Hardware Tier Profiles for Step 8 & Rubric 2
HARDWARE_TIERS = {
    "Raspberry_Pi_4": {
        "cpu": "Quad-core Cortex-A72 @ 1.5 GHz",
        "ram_mb": 4096,
        "idle_power_w": 2.7,
        "peak_power_w": 6.4,
        "perf_factor": 1.0,      # baseline edge reference
        "fuel_rate": 1e-6         # Joules per Wasm fuel unit
    },
    "Jetson_Xavier_NX": {
        "cpu": "6-core Carmel ARMv8.2 @ 1.4 GHz",
        "ram_mb": 8192,
        "idle_power_w": 4.5,
        "peak_power_w": 15.0,
        "perf_factor": 2.6,      # ~2.6x faster than RPi4
        "fuel_rate": 0.75e-6
    },
    "Desktop_x86_64": {
        "cpu": "Intel Core i7-12700 / AMD Ryzen 7 @ 3.6 GHz",
        "ram_mb": 32768,
        "idle_power_w": 25.0,
        "peak_power_w": 95.0,
        "perf_factor": 6.5,      # ~6.5x faster
        "fuel_rate": 0.45e-6
    }
}

# Experimental Seeds (Five seeds per configuration for statistical validity)
SEEDS = [42, 101, 2024, 777, 999]

# Default Federated Parameters
FL_CONFIG = {
    "rounds": 15,
    "local_epochs": 3,
    "batch_size": 32,
    "lr": 0.001,
    "clients_k": [5, 10, 20],
    "default_k": 10,
    "dirichlet_alphas": [0.1, 0.5, 1.0, None], # None represents IID
    "malicious_rates": [0.0, 0.10, 0.20, 0.30],
    "aggregation_methods": ["FedAvg", "FedProx", "FedNova"],
    "fedprox_mu_sweep": [0.01, 0.1, 0.5, 1.0],
    "default_fedprox_mu": 0.1,
    "feature_selection_k_sweep": [10, 20, 30],
    "default_k_features": 20
}

# DRL Trust Agent (Policy Decision Point) Parameters
DRL_CONFIG = {
    "state_dim": 5,      # [anomaly_prob, trust_score, flow_entropy, prev_action, resource_load]
    "action_dim": 4,     # 0: Allow, 1: Re-authenticate, 2: Throttle, 3: Quarantine
    "hidden_dim": 64,
    "gamma": 0.95,
    "lr": 0.001,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    "memory_capacity": 5000,
    "batch_size": 32,
    "target_update_freq": 10,
    # Reward weights: R = lambda_1 * Detection - lambda_2 * FPR - lambda_3 * Latency + lambda_4 * ThreatDelta
    "lambda_detection": 2.0,
    "lambda_fpr": 1.5,
    "lambda_latency": 0.8,
    "lambda_threat_delta": 1.2
}

# Wasm Policy Enforcement Point (PEP) Simulation Parameters
WASM_CONFIG = {
    "fuel_budget_per_inference": 1_000_000,
    "base_instruction_fuel": 10,
    "sandbox_overhead_pct": 18.5, # 18.5% execution time overhead vs native (from Weber et al. / Zhang et al.)
    "wasm_binary_size_kb": 142.5, # Rust compiled wasm32-unknown-unknown opt-level 3
    "native_binary_size_kb": 890.0,
    "wasm_memory_overhead_mb": 4.2 # Sandboxed memory boundary
}

# Cryptographic Layer (Layer 4) Parameters
CRYPTO_CONFIG = {
    "payload_cipher": "AES-256-EAX",
    "signature_algorithm": "RSA-2048-PSS",
    "hash_algorithm": "SHA-256",
    "key_size_bits": 2048,
    "nonce_bytes": 16
}
