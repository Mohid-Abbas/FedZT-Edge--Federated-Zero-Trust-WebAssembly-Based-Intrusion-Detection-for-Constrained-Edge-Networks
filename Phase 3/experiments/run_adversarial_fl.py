"""
Experiment 2: Adversarial Robustness and Update Filter Evaluation (RQ2 & Djemaa Criteria)
Evaluates label-flipping poisoning attacks across malicious client rates (0%, 10%, 20%, 30%)
with and without GMM + K-Means statistical filtering.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import SEEDS, PROCESSED_DATA_DIR, RAW_LOGS_DIR, TABLES_DIR
from src.baselines import run_federated_pipeline

os.makedirs(RAW_LOGS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

def load_data(dataset_name="unsw_nb15", k=20):
    path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}_k{k}.joblib")
    return joblib.load(path)

def run_adversarial_suite(dataset_name="unsw_nb15", num_seeds=3, rounds=8):
    print(f"\n========================================================")
    print(f" Running Adversarial Robustness Suite for [{dataset_name.upper()}]")
    print(f"========================================================")
    
    pkg = load_data(dataset_name, k=20)
    X_train, y_train = pkg["X_train"], pkg["y_train"]
    X_test, y_test = pkg["X_test"], pkg["y_test"]
    
    seeds = SEEDS[:num_seeds]
    malicious_rates = [0.0, 0.10, 0.20, 0.30]
    
    modes = [
        ("No_Defense", "FedAvg", False, False),
        ("FedProx_No_Defense", "FedProx", False, False),
        ("FLTrust", "FLTrust", False, False),
        ("With_GMM_Defense", "FedProx", True, True)
    ]
    
    results = []
    summary_map = {
        "rates": malicious_rates,
        "FedZT_Edge": [],
        "B2_no_defense": [],
        "B3_FedProx": [],
        "FLTrust": []
    }
    
    for mal_rate in malicious_rates:
        print(f"\n>>> Malicious Client Rate: {int(mal_rate * 100)}% <<<")
        for def_str, agg_method, defense_enabled, use_drl in modes:
            print(f"  Configuration: {def_str} ({agg_method})")
            
            f1_runs = []
            prec_runs = []
            rec_runs = []
            benign_pres_runs = []
            
            for seed in seeds:
                res = run_federated_pipeline(
                    X_train, y_train, X_test, y_test,
                    config_id=f"Attack_{int(mal_rate*100)}_{def_str}",
                    num_clients=10,
                    alpha=0.5,
                    rounds=rounds,
                    aggregation_method=agg_method,
                    fedprox_mu=0.1,
                    use_defense=defense_enabled,
                    malicious_rate=mal_rate,
                    use_drl=use_drl,
                    use_crypto=True,
                    seed=seed
                )
                f1_runs.append(res["f1"])
                filt_m = res["filter_metrics_last_round"]
                if filt_m and defense_enabled:
                    prec_runs.append(filt_m.get("filter_precision", 1.0))
                    rec_runs.append(filt_m.get("filter_recall", 1.0))
                    benign_pres_runs.append(filt_m.get("benign_preservation", 1.0))
                    
            f1_mean = round(float(np.mean(f1_runs)), 4)
            f1_std = round(float(np.std(f1_runs)), 4)
            
            if def_str == "With_GMM_Defense":
                summary_map["FedZT_Edge"].append(f1_mean)
            elif def_str == "No_Defense":
                summary_map["B2_no_defense"].append(f1_mean)
            elif def_str == "FedProx_No_Defense":
                summary_map["B3_FedProx"].append(f1_mean)
            elif def_str == "FLTrust":
                summary_map["FLTrust"].append(f1_mean)
                
            record = {
                "dataset": dataset_name,
                "malicious_rate": mal_rate,
                "malicious_pct": f"{int(mal_rate*100)}%",
                "defense_enabled": defense_enabled,
                "defense_mode": def_str,
                "aggregation_method": agg_method,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "filter_precision": round(float(np.mean(prec_runs)), 4) if prec_runs else (1.0 if mal_rate == 0 else 0.0),
                "filter_recall": round(float(np.mean(rec_runs)), 4) if rec_runs else 0.0,
                "benign_preservation": round(float(np.mean(benign_pres_runs)), 4) if benign_pres_runs else 1.0
            }
            results.append(record)
            
    # Compute Delta F1 drop relative to 0% malicious for each mode
    for def_str, _, _, _ in modes:
        clean_runs = [r["f1_mean"] for r in results if r["malicious_rate"] == 0.0 and r["defense_mode"] == def_str]
        clean_ref = clean_runs[0] if clean_runs else 0.85
        for r in results:
            if r["defense_mode"] == def_str:
                r["delta_f1_drop"] = round(clean_ref - r["f1_mean"], 4)
        
    df = pd.DataFrame(results)
    table_csv = os.path.join(TABLES_DIR, f"adversarial_robustness_{dataset_name}.csv")
    df.to_csv(table_csv, index=False)
    
    log_json = os.path.join(RAW_LOGS_DIR, f"adversarial_{dataset_name}.json")
    with open(log_json, "w") as f:
        json.dump(results, f, indent=2)
        
    # Also save consolidated adversarial_robustness.csv and .json
    if dataset_name == "unsw_nb15":
        consolidated_csv = os.path.join(TABLES_DIR, "adversarial_robustness.csv")
        legacy_df = pd.DataFrame({
            "malicious_rate": summary_map["rates"],
            "FedZT_Edge": summary_map["FedZT_Edge"],
            "B2_no_defense": summary_map["B2_no_defense"],
            "B3_FedProx": summary_map["B3_FedProx"],
            "FLTrust": summary_map["FLTrust"]
        })
        legacy_df.to_csv(consolidated_csv, index=False)
        
        consolidated_json = os.path.join(RAW_LOGS_DIR, "adversarial_robustness.json")
        with open(consolidated_json, "w") as f:
            json.dump(summary_map, f, indent=2)
            
    print(f"\nSaved adversarial table to {table_csv}")
    print(df[["malicious_pct", "defense_mode", "f1_mean", "delta_f1_drop", "benign_preservation"]].to_string(index=False))
    return df

if __name__ == "__main__":
    run_adversarial_suite("unsw_nb15", num_seeds=3, rounds=8)
    run_adversarial_suite("cicids2017", num_seeds=3, rounds=8)
