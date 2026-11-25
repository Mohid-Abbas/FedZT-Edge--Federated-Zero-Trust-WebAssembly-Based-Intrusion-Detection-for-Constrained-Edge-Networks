"""
Generate publication-quality PNG figures for FedZT-Edge Phase 4 paper.
100% synchronized with Phase 3 Measured Empirical Evaluation Tables and Logs.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = r"d:\Semester 5\Computer Networks\Research Project\Phase 4"
BASE_DIR = r"d:\Semester 5\Computer Networks\Research Project\Phase 3\results\raw_logs"

C = {
    "fedzt":   "#2166AC",
    "fedavg":  "#D7191C",
    "fedprox": "#F1A340",
    "fednova": "#7B2D8B",
    "fltrust": "#1A9641",
    "b6":      "#808080",
    "b1":      "#000000",
}

matplotlib.rc("font", family="DejaVu Sans", size=10)
matplotlib.rc("axes", titlesize=11, labelsize=10)
matplotlib.rc("xtick", labelsize=9)
matplotlib.rc("ytick", labelsize=9)
matplotlib.rc("legend", fontsize=8.5)

# Load data
with open(os.path.join(BASE_DIR, "adversarial_robustness.json")) as f:
    adv = json.load(f)
with open(os.path.join(BASE_DIR, "heterogeneity_sweep.json")) as f:
    het = json.load(f)
with open(os.path.join(BASE_DIR, "baselines_unsw_nb15.json")) as f:
    bl_u = json.load(f)

# FIG 1: Adversarial Robustness
fig, ax = plt.subplots(figsize=(6, 3.8))
rates = adv["rates"]
ax.plot(rates, adv["FedZT_Edge"],   "o-",  color=C["fedzt"],   lw=2.0, ms=7, label="FedZT-Edge (ours)", zorder=5)
ax.plot(rates, adv["B2_no_defense"],"s--", color=C["fedavg"],  lw=1.6, ms=6, label="FedAvg (no defence)")
ax.plot(rates, adv["B3_FedProx"],   "^-.", color=C["fedprox"], lw=1.6, ms=6, label="FedProx (no defence)")
ax.plot(rates, adv["FLTrust"],      "D:",  color=C["fltrust"], lw=1.6, ms=6, label="FLTrust (needs server data)")

ax.set_xlabel("Fraction of Malicious Clients")
ax.set_ylabel("Macro F1-Score")
ax.set_title("Adversarial Robustness (UNSW-NB15, Label-Flipping)", fontweight="bold")
ax.set_xlim(-0.02, 0.33)
ax.set_ylim(0.20, 0.92)
ax.set_xticks(rates)
ax.set_xticklabels([f"{r:.0%}" for r in rates])
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower left", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_adversarial.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("[OK] fig1_adversarial.png")

# FIG 2: Heterogeneity Sweep
fig, ax = plt.subplots(figsize=(6, 3.8))
alphas = het["alphas"]
ax.semilogx(alphas, het["FedZT_Edge"], "o-",  color=C["fedzt"],   lw=2.0, ms=7, label="FedZT-Edge (ours)", zorder=5)
ax.semilogx(alphas, het["FedAvg"],     "s--", color=C["fedavg"],  lw=1.6, ms=6, label="FedAvg")
ax.semilogx(alphas, het["FedProx"],    "^-.", color=C["fedprox"], lw=1.6, ms=6, label="FedProx")
ax.semilogx(alphas, het["FedNova"],    "D:",  color=C["fednova"], lw=1.6, ms=6, label="FedNova")
ax.set_xlabel("Dirichlet α (lower = more heterogeneous)")
ax.set_ylabel("Macro F1-Score")
ax.set_title("Non-IID Tolerance: F1 vs. Dirichlet α (UNSW-NB15)", fontweight="bold")
ax.set_ylim(0.50, 0.92)
ax.set_xticks(alphas)
ax.set_xticklabels([str(a) for a in alphas])
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax.grid(True, linestyle="--", alpha=0.4, which="both")
ax.legend(loc="lower right", framealpha=0.9)
ax.axvline(x=0.5, color="gray", lw=1, ls="--", alpha=0.5)
ax.text(0.52, 0.52, "Default α=0.5", fontsize=7.5, color="gray", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_heterogeneity.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("[OK] fig2_heterogeneity.png")

# FIG 3: Convergence
def get_avg_history(data, bid):
    hists = [e["round_f1_history"] for e in data if e.get("baseline_id") == bid and "round_f1_history" in e]
    if not hists: return None
    minlen = min(len(h) for h in hists)
    return [np.mean([h[i] for h in hists]) for i in range(minlen)]

fedzt_hist  = get_avg_history(bl_u, "FedZT-Edge")
fedavg_hist = get_avg_history(bl_u, "B2")
fedprox_hist= get_avg_history(bl_u, "B3")
fednova_hist= get_avg_history(bl_u, "B4")

fig, ax = plt.subplots(figsize=(6, 3.8))
def plot_hist(ax, hist, bid, label, color, ls="-", marker="o"):
    if hist:
        rounds = list(range(1, len(hist)+1))
        ax.plot(rounds, hist, f"{marker}{ls}", color=color, lw=2.0 if bid=="FedZT-Edge" else 1.6,
                ms=6 if bid=="FedZT-Edge" else 5, label=label, zorder=5 if bid=="FedZT-Edge" else 3)

plot_hist(ax, fedzt_hist,   "FedZT-Edge", "FedZT-Edge (ours)", C["fedzt"],   "-",  "o")
plot_hist(ax, fedavg_hist,  "B2",         "IID FedAvg",        C["fedavg"],  "--", "s")
plot_hist(ax, fedprox_hist, "B3",         "Non-IID FedAvg",    C["fedprox"], "-.", "^")
plot_hist(ax, fednova_hist, "B4",         "Non-IID FedProx",   C["fednova"], ":",  "D")
ax.set_xlabel("Communication Round")
ax.set_ylabel("Macro F1-Score")
ax.set_title("Convergence over FL Rounds (UNSW-NB15, α=0.5)", fontweight="bold")
ax.set_xlim(0.5, 8.5)
ax.set_ylim(0.70, 0.91)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower right", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_convergence.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("[OK] fig3_convergence.png")

# FIG 4: Baseline Comparison Bar Chart (100% matched with Phase 3 Measured Report & Table 2)
methods = ["B1\nCentral.", "B2\nIID FedAvg", "B3\nNon-IID Avg", "B4\nFedProx", "B5\n+GMM Filter", "B6\nLocal Wasm", "FedZT-\nEdge"]
f1_unsw = [0.878, 0.867, 0.850, 0.841, 0.841, 0.497, 0.841]
f1_cic  = [0.969, 0.944, 0.937, 0.917, 0.917, 0.561, 0.915]

x = np.arange(len(methods))
width = 0.36

fig, ax = plt.subplots(figsize=(7.8, 4.0))
bars1 = ax.bar(x - width/2, f1_unsw, width, label="UNSW-NB15",   color="#2166AC", alpha=0.88, edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, f1_cic,  width, label="CICIDS-2017", color="#D7191C", alpha=0.88, edgecolor="white", linewidth=0.5)

# Highlight FedZT-Edge bars
ax.bar(x[-1] - width/2, f1_unsw[-1], width, color="#1047A9", edgecolor="#FFD700", linewidth=1.8, alpha=0.95)
ax.bar(x[-1] + width/2, f1_cic[-1],  width, color="#A80000", edgecolor="#FFD700", linewidth=1.8, alpha=0.95)

ax.set_xlabel("Method")
ax.set_ylabel("Macro F1-Score")
ax.set_title("Measured Baseline Comparison (Macro F1 on UNSW-NB15 and CICIDS-2017)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylim(0.40, 1.05)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="lower right", framealpha=0.9)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=90)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_baseline_bar.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("[OK] fig4_baseline_bar.png")

# FIG 5: Energy Model
devices   = ["Raspberry Pi 4\n(5W TDP)", "Jetson Xavier NX\n(10W TDP)", "Desktop x86-64\n(65W TDP)"]
infer_mj  = [0.42, 0.18, 0.09]
train_mj  = [8.4,  3.6,  1.8]
crypto_ms = [2.1,  0.9,  0.4]

x = np.arange(len(devices))
width = 0.26

fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
ax2 = ax1.twinx()
b1 = ax1.bar(x - width, infer_mj, width, label="Inference energy (mJ)", color="#2166AC", alpha=0.85)
b2 = ax1.bar(x,          train_mj, width, label="Training/round energy (mJ)", color="#5AAE61", alpha=0.85)
b3 = ax2.bar(x + width, crypto_ms, width, label="Crypto overhead (ms)", color="#F1A340", alpha=0.85)
ax1.set_xlabel("Hardware Tier")
ax1.set_ylabel("Energy (mJ)", color="#2166AC")
ax2.set_ylabel("Crypto Overhead (ms)", color="#F1A340")
ax1.set_title("Multi-Tier Resource Model (Analytical)", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(devices, fontsize=9)
ax1.tick_params(axis="y", labelcolor="#2166AC")
ax2.tick_params(axis="y", labelcolor="#F1A340")
ax1.set_ylim(0, 11)
ax2.set_ylim(0, 3.2)
ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9, fontsize=8)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax1.text(bar.get_x()+bar.get_width()/2., h+0.08, f"{h}", ha="center", va="bottom", fontsize=8)
for bar in b3:
    h = bar.get_height()
    ax2.text(bar.get_x()+bar.get_width()/2., h+0.04, f"{h}", ha="center", va="bottom", fontsize=8, color="#F1A340")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_energy.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("[OK] fig5_energy.png")
