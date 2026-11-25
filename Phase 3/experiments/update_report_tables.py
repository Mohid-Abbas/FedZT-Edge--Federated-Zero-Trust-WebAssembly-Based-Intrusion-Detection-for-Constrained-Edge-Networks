"""
Post-Experiment Report Table Updater
=====================================
Reads the live CSVs produced by run_adversarial_fl.py and run_ablations.py
and generates a fully filled-in LaTeX snippet for Tables 2 (adversarial) and 3 (ablations).

Run AFTER all experiments complete:
    python experiments/update_report_tables.py

Output: results/tables/report_tables_filled.tex
        (paste these table bodies into the main report, or use \input{})
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import TABLES_DIR, RAW_LOGS_DIR, REPORT_DIR

os.makedirs(TABLES_DIR, exist_ok=True)

def fmt(val, decimals=3):
    """Format a numeric value to N decimals."""
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)

def pct(val):
    """Format as percentage string."""
    try:
        return f"{float(val)*100:.1f}\\%"
    except (TypeError, ValueError):
        return str(val)

def build_adversarial_table():
    """Build LaTeX table body for adversarial robustness results."""
    adv_csv = os.path.join(TABLES_DIR, "adversarial_robustness_unsw_nb15.csv")
    if not os.path.exists(adv_csv):
        print(f"[WARN] {adv_csv} not found — adversarial experiment may still be running.")
        return None

    df = pd.read_csv(adv_csv)
    rows = []

    mode_labels = {
        "No_Defense": "FedAvg (No Defence)",
        "FedProx_No_Defense": "FedProx (No Defence)",
        "FLTrust": "FLTrust (Anchor)",
        "With_GMM_Defense": "FedZT-Edge (GMM Filter)"
    }

    malicious_rates = sorted(df["malicious_pct"].unique(), key=lambda s: int(s.replace("%","")))

    lines = []
    for pct_str in malicious_rates:
        sub = df[df["malicious_pct"] == pct_str]
        first = True
        for mode, label in mode_labels.items():
            row = sub[sub["defense_mode"] == mode]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            f1 = fmt(row["f1_mean"])
            delta = fmt(row.get("delta_f1_drop", 0.0))
            bp = pct(row.get("benign_preservation", 1.0)) if row.get("defense_mode") == "With_GMM_Defense" else "---"
            fr = pct(row.get("filter_recall", 0.0)) if row.get("defense_mode") == "With_GMM_Defense" else "---"
            prefix = f"\\addlinespace[2pt]\n${pct_str.replace('%', '\\%')}$" if first else "          "
            first = False
            lines.append(f"${pct_str.replace('%', '\\%')}$ & {label} & ${f1}$ & ${delta}$ & {bp} & {fr} \\\\")

    return "\n".join(lines)


def build_ablation_table():
    """Build LaTeX table body for ablation results."""
    abl_csv = os.path.join(TABLES_DIR, "ablations_summary_unsw_nb15.csv")
    if not os.path.exists(abl_csv):
        print(f"[WARN] {abl_csv} not found — ablation experiment may still be running.")
        return None

    df = pd.read_csv(abl_csv)
    lines = []

    for _, row in df.iterrows():
        ab_id = row.get("ablation_id", "?")
        name = row.get("name", "?")
        variant = row.get("variant", "?")
        f1m = fmt(row.get("f1_mean", 0.0))
        f1s = fmt(row.get("f1_std", 0.0))
        notes = str(row.get("notes", "")).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
        lines.append(f"({ab_id}) & {name} & {variant} & ${f1m} \\pm {f1s}$ & {notes} \\\\")

    return "\n".join(lines)


def build_summary_text(adv_csv_path):
    """Generate narrative summary of adversarial results for the report discussion."""
    if not os.path.exists(adv_csv_path):
        return ""

    df = pd.read_csv(adv_csv_path)
    gmm = df[df["defense_mode"] == "With_GMM_Defense"]
    no_def = df[df["defense_mode"] == "No_Defense"]

    rows = []
    for pct_str in ["20%", "30%"]:
        g = gmm[gmm["malicious_pct"] == pct_str]
        n = no_def[no_def["malicious_pct"] == pct_str]
        if len(g) > 0 and len(n) > 0:
            g_f1 = g.iloc[0]["f1_mean"]
            n_f1 = n.iloc[0]["f1_mean"]
            delta_no = n.iloc[0].get("delta_f1_drop", 0.0)
            delta_gmm = g.iloc[0].get("delta_f1_drop", 0.0)
            bp = g.iloc[0].get("benign_preservation", 1.0) * 100.0
            fr = g.iloc[0].get("filter_recall", 0.0) * 100.0
            mitigation_pct = (1.0 - delta_gmm / max(delta_no, 1e-6)) * 100.0 if delta_no > 0 else 100.0
            rows.append(f"  - At {pct_str} attack: undefended F1={n_f1:.3f} (drop={delta_no:.3f}), "
                        f"GMM-defended F1={g_f1:.3f} (drop={delta_gmm:.3f}), "
                        f"mitigation={mitigation_pct:.0f}%, "
                        f"benign preservation={bp:.1f}%, filter recall={fr:.1f}%")
    return "\n".join(rows)


def main():
    print("=== Updating Report Tables from Live Experiment CSVs ===\n")

    adv_csv = os.path.join(TABLES_DIR, "adversarial_robustness_unsw_nb15.csv")
    abl_csv = os.path.join(TABLES_DIR, "ablations_summary_unsw_nb15.csv")

    adv_body = build_adversarial_table()
    abl_body = build_ablation_table()
    adv_summary = build_summary_text(adv_csv)

    output_path = os.path.join(TABLES_DIR, "report_tables_filled.tex")
    with open(output_path, "w") as f:
        f.write("% ===== ADVERSARIAL TABLE BODY (paste into Table 2) =====\n")
        if adv_body:
            f.write(adv_body + "\n")
        else:
            f.write("% [NOT YET AVAILABLE — run experiments/run_adversarial_fl.py first]\n")

        f.write("\n% ===== ABLATION TABLE BODY (paste into Table 3) =====\n")
        if abl_body:
            f.write(abl_body + "\n")
        else:
            f.write("% [NOT YET AVAILABLE — run experiments/run_ablations.py first]\n")

        f.write("\n% ===== NARRATIVE SUMMARY (for Section 5 discussion) =====\n")
        if adv_summary:
            f.write("% " + adv_summary.replace("\n", "\n% ") + "\n")
        else:
            f.write("% [NOT YET AVAILABLE]\n")

    print(f"Saved filled table snippets to: {output_path}")

    # Also print key stats to console
    if adv_body:
        print("\n--- Adversarial Table (UNSW-NB15) ---")
        print(adv_body)
    if adv_summary:
        print("\n--- Adversarial Narrative ---")
        print(adv_summary)
    if abl_body:
        print("\n--- Ablation Table ---")
        print(abl_body)


if __name__ == "__main__":
    main()
