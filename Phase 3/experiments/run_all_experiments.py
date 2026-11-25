"""
Master Experiment Runner for FedZT-Edge (Phase 3)
Sequentially executes Baselines, Adversarial FL, Heterogeneity,
Ablations, and Visualizations, ensuring complete end-to-end reproducibility.
"""

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_baselines import run_baselines_suite
from experiments.run_adversarial_fl import run_adversarial_suite
from experiments.run_heterogeneity import run_heterogeneity_suite
from experiments.run_ablations import run_all_ablations
from experiments.visualize_results import main as generate_visualizations

def run_all(num_seeds=3, rounds=8):
    total_start = time.perf_counter()
    print("======================================================================")
    print(" Starting FedZT-Edge Comprehensive Experimental Evaluation Suite")
    print(f" Datasets: UNSW-NB15 & CICIDS2017 | Communication Rounds: {rounds} | Seeds: {num_seeds}")
    print("======================================================================\n")
    
    # 1. Baseline Evaluation (B1 to B6 vs FedZT-Edge Treatment)
    print("\n[STEP 1/5] RUNNING BASELINE EVALUATION...")
    run_baselines_suite("unsw_nb15", num_seeds=num_seeds, rounds=rounds)
    run_baselines_suite("cicids2017", num_seeds=num_seeds, rounds=rounds)
    
    # 2. Adversarial Robustness under Label Flipping (RQ2 & Djemaa Criteria)
    print("\n[STEP 2/5] RUNNING ADVERSARIAL POISONING EXPERIMENTS...")
    run_adversarial_suite("unsw_nb15", num_seeds=num_seeds, rounds=rounds)
    run_adversarial_suite("cicids2017", num_seeds=num_seeds, rounds=rounds)
    
    # 3. Non-IID Dirichlet Skew (RQ1)
    print("\n[STEP 3/5] RUNNING HETEROGENEITY EXPERIMENTS...")
    run_heterogeneity_suite("unsw_nb15", num_seeds=num_seeds, rounds=rounds)
    run_heterogeneity_suite("cicids2017", num_seeds=num_seeds, rounds=rounds)
    
    # 4. Ablation Studies (a through h)
    print("\n[STEP 4/5] RUNNING ALL 8 ABLATION STUDIES (a--h)...")
    run_all_ablations("unsw_nb15", num_seeds=num_seeds, rounds=rounds)
    run_all_ablations("cicids2017", num_seeds=num_seeds, rounds=rounds)
    
    # 5. Visualizations & Graphical Results
    print("\n[STEP 5/5] GENERATING PUBLICATION-GRADE VISUALIZATIONS...")
    generate_visualizations()
    
    elapsed = time.perf_counter() - total_start
    print("\n======================================================================")
    print(f" All Experimental Suites Completed Successfully in {elapsed:.1f}s!")
    print(" Visualizations: Phase 3/results/figures/")
    print(" Tables:         Phase 3/results/tables/")
    print(" Raw Logs:       Phase 3/results/raw_logs/")
    print("======================================================================")

if __name__ == "__main__":
    run_all(num_seeds=3, rounds=8)
