"""
Statistical Significance Tests (Step 9)
Runs Welch t-test / Wilcoxon, Holm-Bonferroni correction, and Cohen's d
across all pairwise comparisons between FedZT-Edge and B1-B6 baselines.
Saves a p-value table CSV to results/tables/statistical_tests.csv.
"""

import os, sys, json
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.statistical_tests import compare_configurations_statistically, holm_bonferroni_correction
from src.config import TABLES_DIR, RAW_LOGS_DIR

os.makedirs(TABLES_DIR, exist_ok=True)

def load_per_seed_f1(dataset_name="unsw_nb15"):
    path = os.path.join(RAW_LOGS_DIR, f"baselines_{dataset_name}.json")
    with open(path) as f:
        raw = json.load(f)
    # Group per baseline_id across seeds
    groups = {}
    for rec in raw:
        bid = rec.get("baseline_id") or rec.get("config_id", "FedZT-Edge")
        f1  = rec.get("f1", rec.get("f1_score", None))
        if f1 is not None:
            groups.setdefault(bid, []).append(float(f1))
    return groups

def run_stats(dataset_name="unsw_nb15"):
    print(f"\n=== Statistical Tests: {dataset_name.upper()} ===")
    try:
        groups = load_per_seed_f1(dataset_name)
    except Exception as e:
        print(f"  Could not load raw logs: {e}")
        return []

    fedzt = groups.get("FedZT-Edge", groups.get("FedZT", []))
    if not fedzt:
        print("  FedZT-Edge samples not found in log — skipping.")
        return []

    comparisons = []
    for bid, samples in groups.items():
        if bid in ("FedZT-Edge", "FedZT"):
            continue
        result = compare_configurations_statistically(fedzt, samples,
                                                       config_name=f"FedZT-Edge vs {bid}")
        result["dataset"] = dataset_name
        comparisons.append(result)

    # Holm-Bonferroni correction over all raw p-values
    raw_ps = [r["raw_p_value"] for r in comparisons]
    adj_ps = holm_bonferroni_correction(raw_ps)
    for r, adj in zip(comparisons, adj_ps):
        r["holm_adjusted_p"] = round(adj, 6)
        r["significant_alpha05"] = adj < 0.05

    for r in comparisons:
        sig = "YES" if r["significant_alpha05"] else "no"
        print(f"  {r['comparison']:40s}  p={r['raw_p_value']:.4f}  p_holm={r['holm_adjusted_p']:.4f}  d={r['cohens_d']:+.3f}  sig={sig}")

    return comparisons

results = []
for ds in ["unsw_nb15", "cicids2017"]:
    results.extend(run_stats(ds))

if results:
    df = pd.DataFrame(results)[[
        "dataset", "comparison", "mean_a", "std_a", "mean_b", "std_b",
        "test_type", "statistic", "raw_p_value", "holm_adjusted_p",
        "cohens_d", "significant_alpha05"
    ]]
    out = os.path.join(TABLES_DIR, "statistical_tests.csv")
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved statistical tests table to: {out}")
    print(df.to_string(index=False))
else:
    print("\nNo results to save — raw logs may be missing per-seed records.")
    print("Note: With 3 seeds, Mann-Whitney U cannot reach p<0.05 (minimum p=0.100).")
    print("Statistical power is acknowledged as a limitation in the report.")
