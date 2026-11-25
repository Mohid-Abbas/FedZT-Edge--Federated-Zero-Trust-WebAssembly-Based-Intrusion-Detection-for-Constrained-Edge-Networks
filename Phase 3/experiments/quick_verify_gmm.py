"""
Quick GMM Filter Verification
Runs just 2 attack rates (0% and 20%) with 2 seeds to verify the fixed
cosine-distance GMM filter actually catches poisoned updates.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import joblib
from src.config import SEEDS, PROCESSED_DATA_DIR
from src.baselines import run_federated_pipeline

pkg = joblib.load(os.path.join(PROCESSED_DATA_DIR, "unsw_nb15_k20.joblib"))
X_train, y_train = pkg["X_train"], pkg["y_train"]
X_test, y_test   = pkg["X_test"],  pkg["y_test"]

print("=== Quick GMM Filter Verification (2 seeds, rates 0% and 20%) ===")
for mal_rate in [0.0, 0.20]:
    for defense in [False, True]:
        f1s, recalls, bps = [], [], []
        for seed in SEEDS[:2]:
            res = run_federated_pipeline(
                X_train, y_train, X_test, y_test,
                config_id=f"gmm_verify_{mal_rate}_{defense}",
                num_clients=10, alpha=0.5, rounds=8,
                aggregation_method="FedProx", fedprox_mu=0.1,
                use_defense=defense, malicious_rate=mal_rate,
                use_drl=False, use_crypto=False, seed=seed
            )
            f1s.append(res["f1"])
            fm = res.get("filter_metrics_last_round", {}) or {}
            recalls.append(fm.get("filter_recall", 0.0))
            bps.append(fm.get("benign_preservation", 1.0))
        tag = "GMM ON " if defense else "No Defense"
        print(f"  mal={int(mal_rate*100):2d}% | {tag} | F1={np.mean(f1s):.4f} | "
              f"filter_recall={np.mean(recalls):.3f} | benign_pres={np.mean(bps):.3f}")

print("\nDone.")
