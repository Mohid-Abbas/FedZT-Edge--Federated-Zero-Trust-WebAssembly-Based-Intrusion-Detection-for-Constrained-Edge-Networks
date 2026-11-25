# FedZT-Edge: Phase 3 Research Implementation & Experimental Evaluation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Fast-NUCES](https://img.shields.io/badge/FAST--NUCES-CS--Fall2026-darkblue.svg)]()

This repository contains the complete implementation, experimental evaluation suite, baseline comparisons, ablation studies, and publication-grade visualizations for **FedZT-Edge**: A Federated, Zero-Trust, WebAssembly-Based Intrusion Detection Framework for Resource-Constrained Edge Networks.

> **Reproducibility Notice:** All results in `results/tables/` and `results/raw_logs/` are generated exclusively by running the experiment scripts below. No values are hard-coded or manually adjusted. The WebAssembly execution layer (Layer 1) is **simulated in Python** using a parameterized overhead model (18.5% latency factor from `WASM_CONFIG`); a physical Wasmtime runtime is not required. Energy figures are analytically modelled from hardware specification constants, not measured on physical hardware.

---

## 📋 Table of Contents
1. [Project Overview & Architecture](#-project-overview--architecture)
2. [Folder Structure](#-folder-structure)
3. [Setup & Installation](#-setup--installation)
4. [Reproducing All Experiments](#-reproducing-all-experiments)
5. [Five-Layer Framework Implementation](#-five-layer-framework-implementation)
6. [Baselines Evaluated (B1 to B6 vs FedZT-Edge)](#-baselines-evaluated-b1-to-b6-vs-fedzt-edge)
7. [Ablation Studies (a through h)](#-ablation-studies-a-through-h)
8. [Generated Figures & Tables](#-generated-figures--tables)
9. [Report Compilation](#-report-compilation)

---

## 🏛 Project Overview & Architecture

FedZT-Edge bridges the gap between federated intrusion detection, adaptive Zero-Trust policy decisions, and lightweight, isolated WebAssembly edge execution. It is engineered as five cooperating layers:

- **Layer 1 — WebAssembly Policy Enforcement Points (PEPs):** Sandboxed edge inference compiled with fuel metering, memory bounds, latency profiling, and edge local policy snapshot fallback for offline partition tolerance.
- **Layer 2 — Federated Learning with Poisoning Defence:** Federated coordination supporting FedAvg, FedProx (with proximal constraint $\mu$), and FedNova under Dirichlet non-IID class skew ($\alpha \in \{0.1, 0.5, 1.0, \infty\}$) protected by a GMM + K-Means statistical update filter defending against 0–30% malicious label-flipping poisoning attacks.
- **Layer 3 — Deep Reinforcement Learning Trust Agent (PDP):** A Deep Q-Network (DQN) policy decision point operating over state $S_t = [\text{anomaly\_prob}, \text{trust\_score}, \text{flow\_entropy}, \text{prev\_action}, \text{resource\_load}]$, selecting actions from $\{\text{Allow}, \text{Adaptive Re-auth}, \text{Throttle}, \text{Quarantine}\}$ using a multi-objective reward balancing detection utility, false-positive penalties, and latency overhead.
- **Layer 4 — Cryptographic Protection Layer:** Cross-cutting envelope employing AES-256 (EAX/GCM) payload encryption, RSA-2048 digital signatures, and SHA-256 integrity digests.
- **Layer 5 — Measurement Infrastructure:** Real-time instrumentation tracking mean/p95 latency, resident memory (RAM), Wasm fuel consumption, binary size, and per-round energy dissipation across three hardware tiers:
  1. **Raspberry Pi 4** (Quad Cortex-A72 @ 1.5 GHz, 4GB RAM)
  2. **Jetson Xavier NX** (6-core Carmel ARMv8.2, 8GB RAM)
  3. **Desktop x86-64** (Intel Core i7 / AMD Ryzen, 32GB RAM)

---

## 📁 Folder Structure

```
Phase 3/
├── data/
│   ├── download_and_prepare_datasets.py  # Automated acquisition with MD5 checksums
│   ├── data_manifest.json                # Versioned data lake manifest
│   ├── raw/                              # Raw parquets & CSVs (UNSW-NB15, CICIDS2017)
│   └── processed/                        # Preprocessed, balanced, Chi2-selected joblib matrices
├── src/
│   ├── __init__.py
│   ├── config.py                         # Hyperparameters, hardware profiles, seeds
│   ├── preprocessing.py                  # Cleaning, Chi2 feature ranking, Dirichlet non-IID shards
│   ├── layer1_wasm_pep.py                # Layer 1: Wasm PEP, fuel metering, fallback snapshots
│   ├── layer2_federated.py               # Layer 2: FL algorithms (FedAvg/Prox/Nova) + GMM filter
│   ├── layer3_drl_pdp.py                 # Layer 3: DQN Trust Agent + Static Threshold policy
│   ├── layer4_crypto.py                  # Layer 4: AES-256, RSA-2048, SHA-256 envelope
│   ├── layer5_metrics.py                 # Layer 5: Latency, fuel, memory & multi-tier energy
│   ├── baselines.py                      # B1-B6 baseline and treatment runner
│   └── statistical_tests.py              # Shapiro-Wilk, Welch t-test, Wilcoxon, Holm, Cohen's d
├── experiments/
│   ├── run_baselines.py                  # Runs B1 through B6 vs FedZT-Edge
│   ├── run_adversarial_fl.py             # Runs 0-30% label-flipping poisoning sweeps
│   ├── run_heterogeneity.py              # Runs Dirichlet non-IID alpha sweeps
│   ├── run_ablations.py                  # Runs all 8 ablation studies (a through h)
│   ├── run_all_experiments.py            # Master benchmark execution runner
│   └── visualize_results.py              # Generates publication-grade 300 DPI figures
├── tests/
│   ├── test_crypto.py                    # Unit tests for AES/RSA/SHA-256
│   ├── test_wasm_pep.py                  # Unit tests for Wasm PEP & fuel accounting
│   ├── test_gmm_filter.py                # Unit tests for GMM defense & FL aggregation
│   ├── test_drl_agent.py                 # Unit tests for DQN action & reward logic
│   └── test_end_to_end.py                # Integration tests for full pipeline
├── results/
│   ├── figures/                          # 300-DPI charts (PNG/PDF)
│   ├── tables/                           # CSV summary tables of experimental results
│   └── raw_logs/                         # JSON execution logs for byte-for-byte reproducibility
├── report/
│   └── Phase3_FedZT-Edge_Implementation_Report.tex  # Complete Overleaf-ready LaTeX Report
├── requirements.txt
├── run_phase3.bat                        # 1-Click Windows execution script
├── run_phase3.ps1                        # 1-Click PowerShell execution script
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Requirements
- Python 3.10 or higher
- NVIDIA CUDA (optional for GPU acceleration, CPU execution fully supported)

### 2. Environment Setup
```bash
# Clone/navigate to Phase 3 directory
cd "Phase 3"

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run Unit Tests (100% Passing Verification)
```bash
python -m unittest discover -s tests
```

---

## ⚡ Reproducing All Experiments

You can reproduce the entire experimental evaluation suite step-by-step:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run unit tests (must all pass)
python -m unittest discover -s tests

# Step 3: Run baselines (B1–B6 vs FedZT-Edge), ~15 min
python experiments/run_baselines.py

# Step 4: Run adversarial robustness suite (4 modes × 4 rates × 3 seeds × 2 datasets), ~45–90 min
python experiments/run_adversarial_fl.py

# Step 5: Run heterogeneity sweep (4 methods × 4 alpha × 3 seeds × 2 datasets), ~30–60 min
python experiments/run_heterogeneity.py

# Step 6: Run all 8 ablation studies (a–h), ~30–60 min
python experiments/run_ablations.py

# Step 7: Generate all 6 figures from live CSVs
python experiments/visualize_results.py

# Step 8 (optional): Fill report table snippets from live CSVs
python experiments/update_report_tables.py
```

Or run the master script which calls all of the above:
```bash
python experiments/run_all_experiments.py
```

---

## 📊 Baselines Evaluated (B1 to B6 vs FedZT-Edge)

| Baseline ID | Description | Primary Isolation Target |
|:---|:---|:---|
| **B1** | Centralized training on pooled data | Theoretical performance upper bound |
| **B2** | IID Federated FedAvg without defence | Traditional FL reference |
| **B3** | Non-IID Federated FedAvg without defence | Heterogeneity penalty under Dirichlet skew (RQ1) |
| **B4** | Non-IID FedProx ($\mu = 0.1$) | Proximal term stabilization under skew |
| **B5** | Non-IID FL + GMM Update Filtering | Statistical defence against poisoning (RQ2) |
| **B6** | Local Wasm-only IDS on each device | Benefit of cooperation vs device isolation |
| **FedZT-Edge** | Full 5-Layer Stack (FedProx + GMM + DQN + Wasm + Crypto) | Unified proposed treatment condition |

---

## 🔬 Ablation Studies (a through h)

All 8 ablation studies specified in Deliverable D3 Step 10 are implemented and validated:

1. **Ablation (a): Update Filter:** Disabled vs Enabled under 0%, 10%, 20%, 30% label flipping.
2. **Ablation (b): FedProx Regularization Sweep:** $\mu \in \{0.01, 0.1, 0.5, 1.0\}$ under severe skew ($\alpha = 0.1$).
3. **Ablation (c): Feature Count Sweep:** Top Chi-square features $k \in \{10, 20, 30\}$.
4. **Ablation (d): DRL Agent vs Static Rules:** DQN dynamic trust agent vs deterministic threshold policy.
5. **Ablation (e): Cryptographic Envelope Overhead:** AES-256 + RSA-2048 enabled vs disabled.
6. **Ablation (f): Wasm vs Native Execution:** Sandboxed Wasmtime fuel economics vs native execution.
7. **Ablation (g): Client Churn & Stragglers:** 20% random client dropout vs stable network.
8. **Ablation (h): Zero-Trust Availability:** PDP online agent vs local edge fallback snapshot.

---

## 📈 Generated Figures & Tables

All figures are automatically generated in `results/figures/` at 300 DPI:
- `fig1_baseline_comparison.png`: Macro F1 and energy footprint across B1–B6 and FedZT-Edge.
- `fig2_heterogeneity_convergence.png`: Communication rounds convergence across Dirichlet $\alpha \in \{0.1, 0.5, 1.0, \infty\}$.
- `fig3_adversarial_robustness.png`: Poisoning resilience & Djemaa benign preservation rate under 0–30% attacks.
- `fig4_pareto_frontiers.png`: Multi-tier Detection vs Energy Pareto curves & Robustness vs Communication curves.
- `fig5_wasm_and_drl.png`: Wasm execution overhead profile & DRL mitigation action allocation.
- `fig6_confusion_matrices.png`: Per-class confusion matrices for UNSW-NB15 and CICIDS2017.

---

## 📝 Report Compilation

The complete academic report is written in Overleaf-ready LaTeX:
- **File:** `Phase 3/report/Phase3_FedZT-Edge_Implementation_Report.tex`
- **Compiler:** `pdfLaTeX` (run twice for cross-references)
- **Formatting:** Self-contained, includes tables, TikZ diagrams, and citations matching Phase 1 & Phase 2.

---
**Authors:** Mohid Abbas (24i-0074), Ali Khan (24i-0069)  
**Institution:** FAST National University of Computer and Emerging Sciences (FAST-NUCES)  
**Course:** Computer Networks (CS Fall 2026)
