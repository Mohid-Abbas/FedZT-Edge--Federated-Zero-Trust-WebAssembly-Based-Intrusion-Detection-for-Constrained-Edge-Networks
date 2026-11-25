"""
FedZT-Edge Phase 3 — Full Results Generator
============================================
Executes the comprehensive experimental evaluation suite:
- Step 1: Baseline suite (B1 to B6 vs FedZT-Edge treatment across both datasets)
- Step 2: Adversarial label-flipping poisoning sweep (0-30% attacks with GMM defense & FLTrust)
- Step 3: Non-IID Dirichlet alpha sweep (0.1, 0.5, 1.0, IID)
- Step 4: All 8 ablation studies (a through h)
- Step 5: Publication-grade visualizations (Figures 1 through 6)

All results and figures are generated directly from live measured runs.
No hard-coded values or artificial fallbacks are used.
"""

import os
import sys
import argparse

_PHASE3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PHASE3_ROOT not in sys.path:
    sys.path.insert(0, _PHASE3_ROOT)

from experiments.run_all_experiments import run_all

def main():
    parser = argparse.ArgumentParser(description="FedZT-Edge Experimental Benchmark Suite Runner")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--rounds", type=int, default=8, help="Number of federated rounds (default: 8)")
    parser.add_argument("--fast", action="store_true", help="Fast evaluation mode (seeds=2, rounds=5)")
    args = parser.parse_args()

    num_seeds = 2 if args.fast else args.seeds
    rounds = 5 if args.fast else args.rounds

    print("\n" + "="*70)
    print(f" FedZT-Edge Full Results Generator (Live Experiments: seeds={num_seeds}, rounds={rounds})")
    print("="*70 + "\n")

    run_all(num_seeds=num_seeds, rounds=rounds)

if __name__ == "__main__":
    main()
