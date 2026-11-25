# Phase 4 — Research Paper Writing & Final Submission
## FedZT-Edge: Federated, Zero-Trust, WebAssembly-Based IDS for Edge Networks

**Authors:** Mohid Abbas (24i-0074) · Ali Khan (24i-0069)  
**Course:** Computer Networks (BS DS/AI) — FAST-NUCES, Fall 2026  
**Due:** November 20, 2026, 11:59 PM | **Points:** 16  
**Official Template:** Springer Nature LaTeX Template (`sn-jnl.cls`)

---

## 🚀 1-STEP OVERLEAF UPLOAD & COMPILE

The zip file **[`Phase4_Overleaf_Package.zip`](file:///d:/Semester%205/Computer%20Networks/Research%20Project/Phase%204/Phase4_Overleaf_Package.zip)** now contains the **official Springer Nature class file (`sn-jnl.cls`)**, all official bibliography styles (`.bst`), the manuscript, and all 5 publication figures.

### Steps:
1. Open **[Overleaf](https://www.overleaf.com/)**.
2. Click **New Project** $\rightarrow$ **Upload Project**.
3. Select and upload **`Phase4_Overleaf_Package.zip`**.
4. Click **Recompile** (Compiler: **pdfLaTeX**).
5. Done! The paper compiles under the official Springer Nature template with zero missing file errors and zero table clipping.

---

## 🛠️ What Was Fixed on Page 5 / 6:
* **Table 3 Auto-Wrapping (`X` column):** Previously, Table 3 used a fixed column width that exceeded the printable boundary, causing text (`attacks`, `updates`, `weigh`, `foo`, `edge`) to be cut off past the right page margin. It has now been replaced with an adaptive `tabularx` using an auto-calculating `X` column (`\begin{tabularx}{\textwidth}{@{}llcX@{}}`), ensuring every cell wraps within the exact text margins without overflowing.
* **Table 4 Single-Column Fit:** Formatted with compact multi-level stacked headers so it fits within the column boundary.
* **Official Springer Nature Class Included:** `sn-jnl.cls` and all 7 associated `.bst` files are bundled directly into the zip package so Overleaf uses the official Springer engine.

---

## 📄 Contents of `Phase4_Overleaf_Package.zip`:
* `main.tex` / `sn-article.tex` — Official Springer Nature manuscript (`\documentclass[sn-mathphys,Numbered]{sn-jnl}`)
* `sn-jnl.cls` — The core official Springer Nature document class
* `sn-mathphys.bst`, `sn-basic.bst`, `sn-nature.bst`, `sn-vancouver.bst`, `sn-aps.bst`, `sn-chicago.bst`, `sn-apacite.bst` — Official reference styles
* `empty.eps` — Official fallback vector image
* `fig1_adversarial.png` — Adversarial robustness plot (0% to 30% label-flipping)
* `fig2_heterogeneity.png` — Dirichlet $\alpha$ non-IID sweep plot
* `fig3_convergence.png` — Empirical multi-round convergence trajectories
* `fig4_baseline_bar.png` — Grouped bar chart comparing all 7 baselines across UNSW-NB15 & CICIDS-2017
* `fig5_energy.png` — Multi-tier resource and latency model plot
