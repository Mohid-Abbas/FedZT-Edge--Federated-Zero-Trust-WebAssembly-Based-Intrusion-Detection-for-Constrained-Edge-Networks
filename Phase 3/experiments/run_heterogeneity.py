"""
Experiment 3: Dirichlet Non-IID Heterogeneity Evaluation (RQ1)
Evaluates FedAvg, FedProx, and FedNova across Dirichlet skews (alpha = 0.1, 0.5, 1.0, IID)
and client counts (K = 5, 10, 20).
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

def run_heterogeneity_suite(dataset_name="unsw_nb15", num_seeds=3, rounds=8):
    print(f"\n========================================================")
    print(f" Running Heterogeneity Suite for [{dataset_name.upper()}]")
    print(f"========================================================")
    
    pkg = load_data(dataset_name, k=20)
    X_train, y_train = pkg["X_train"], pkg["y_train"]
    X_test, y_test = pkg["X_test"], pkg["y_test"]
    
    seeds = SEEDS[:num_seeds]
    alphas = [0.1, 0.5, 1.0, float('inf')]
    methods = [
        ("FedAvg", "FedAvg", False, False),
        ("FedProx", "FedProx", False, False),
        ("FedNova", "FedNova", False, False),
        ("FedZT_Edge", "FedProx", True, True)
    ]
    
    results = []
    sweep_map = {
        "alphas": [a if a != float('inf') else 10.0 for a in alphas],
        "FedZT_Edge": [],
        "FedAvg": [],
        "FedProx": [],
        "FedNova": []
    }
    
    for alpha in alphas:
        alpha_label = "IID" if alpha == float('inf') else f"Dirichlet_a{alpha}"
        print(f"\n>>> Partitioning: {alpha_label} <<<")
        
        for m_name, agg_m, def_on, drl_on in methods:
            print(f"  Method: {m_name}")
            f1_runs = []
            acc_runs = []
            round_curves = []
            
            for seed in seeds:
                res = run_federated_pipeline(
                    X_train, y_train, X_test, y_test,
                    config_id=f"{m_name}_{alpha_label}",
                    num_clients=10,
                    alpha=alpha,
                    rounds=rounds,
                    aggregation_method=agg_m,
                    fedprox_mu=0.1,
                    use_defense=def_on,
                    use_drl=drl_on,
                    use_crypto=False,
                    seed=seed
                )
                f1_runs.append(res["f1"])
                acc_runs.append(res["accuracy"])
                round_curves.append(res["round_f1_history"])
                
            avg_curve = np.mean(round_curves, axis=0).tolist()
            f1_mean = round(float(np.mean(f1_runs)), 4)
            sweep_map[m_name].append(f1_mean)
            
            record = {
                "dataset": dataset_name,
                "alpha_val": alpha if alpha != float('inf') else "inf",
                "alpha_label": alpha_label,
                "method": m_name,
                "f1_mean": f1_mean,
                "f1_std": round(float(np.std(f1_runs)), 4),
                "accuracy_mean": round(float(np.mean(acc_runs)), 4),
                "round_f1_curve": avg_curve
            }
            results.append(record)
            
    df = pd.DataFrame(results)
    table_csv = os.path.join(TABLES_DIR, f"heterogeneity_{dataset_name}.csv")
    df.drop(columns=["round_f1_curve"]).to_csv(table_csv, index=False)
    
    log_json = os.path.join(RAW_LOGS_DIR, f"heterogeneity_{dataset_name}.json")
    with open(log_json, "w") as f:
        json.dump(results, f, indent=2)
        
    # Also save consolidated heterogeneity_sweep.csv and .json
    if dataset_name == "unsw_nb15":
        consolidated_csv = os.path.join(TABLES_DIR, "heterogeneity_sweep.csv")
        legacy_df = pd.DataFrame({
            "alpha": sweep_map["alphas"],
            "FedZT_Edge": sweep_map["FedZT_Edge"],
            "FedAvg": sweep_map["FedAvg"],
            "FedProx": sweep_map["FedProx"],
            "FedNova": sweep_map["FedNova"]
        })
        legacy_df.to_csv(consolidated_csv, index=False)
        
        consolidated_json = os.path.join(RAW_LOGS_DIR, "heterogeneity_sweep.json")
        with open(consolidated_json, "w") as f:
            json.dump(sweep_map, f, indent=2)
        
    print(f"\nSaved heterogeneity table to {table_csv}")
    print(df[["alpha_label", "method", "f1_mean", "accuracy_mean"]].to_string(index=False))
    return df

if __name__ == "__main__":
    run_heterogeneity_suite("unsw_nb15", num_seeds=3, rounds=8)
    run_heterogeneity_suite("cicids2017", num_seeds=3, rounds=8)
