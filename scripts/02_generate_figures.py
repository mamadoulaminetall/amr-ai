"""
AMR-AI — Phase 2: Publication Figures
6 figures for hyper-meta-analysis manuscript
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs("figures", exist_ok=True)

# Dark theme
BG = "#0f172a"
SURFACE = "#1e293b"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
GREEN = "#22c55e"
RED = "#ef4444"
ORANGE = "#f97316"
YELLOW = "#eab308"
PURPLE = "#a855f7"
PINK = "#ec4899"

STRAT_COLORS = {
    "Phage Therapy": "#3b82f6",
    "Antimicrobial Peptides": "#22c55e",
    "CRISPR Antimicrobial": "#a855f7",
    "Synergistic Combinations": "#f97316",
    "De Novo AI Molecules": "#ec4899",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": "#334155",
    "grid.alpha": 0.4, "font.family": "DejaVu Sans",
})

df_studies = pd.read_csv("data/studies_registry.csv")
df_meta = pd.read_csv("data/meta_analytic_estimates.csv")
df_mol = pd.read_csv("data/molecules_db.csv")
df_eskape = pd.read_csv("data/eskape_pathogens.csv")
df_sub = pd.read_csv("data/subgroup_analysis.csv")

total_patients = df_studies["n_patients"].sum()
n_studies = len(df_studies)

# ── Fig 1: Forest Plot — Efficacy by strategy ────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9), facecolor=BG)
ax.set_facecolor(SURFACE)

strategies = df_meta["strategy"].tolist()
y_pos = list(range(len(strategies)))

for i, (_, row) in enumerate(df_meta.iterrows()):
    color = STRAT_COLORS[row["strategy"]]
    ax.plot([row["efficacy_ci_lo"], row["efficacy_ci_hi"]], [i, i],
            color=color, lw=2.5, solid_capstyle="round")
    ax.scatter(row["pooled_efficacy"], i, s=180, color=color, zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.text(row["efficacy_ci_hi"] + 0.01, i,
            f"{row['pooled_efficacy']:.1%}  [{row['efficacy_ci_lo']:.1%}–{row['efficacy_ci_hi']:.1%}]",
            va="center", fontsize=9, color=TEXT)
    ax.text(-0.03, i,
            f"n={row['n_studies']} studies  N={row['total_patients']:,}",
            va="center", ha="right", fontsize=8.5, color=MUTED)

ax.axvline(0.5, color=MUTED, lw=1, ls="--", alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([s.replace(" ", "\n") for s in strategies], fontsize=10)
ax.set_xlim(-0.05, 1.08)
ax.set_xlabel("Pooled Efficacy Rate (Random-Effects, 95% CI)", fontsize=11)
ax.set_title(
    f"Fig 1 — Forest Plot: Efficacy of AMR Alternatives\n"
    f"{n_studies} studies · {total_patients:,} patients · 2015–2026",
    fontsize=13, fontweight="bold", pad=15
)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig1_forest_plot.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig1 ✓")

# ── Fig 2: Pooled rates — Efficacy + Resistance emergence + Mortality reduction
fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
metrics = [
    ("pooled_efficacy", "efficacy_ci_lo", "efficacy_ci_hi", "Pooled Efficacy", GREEN),
    ("pooled_resistance", "resistance_ci_lo", "resistance_ci_hi", "Resistance Emergence", RED),
    ("pooled_mortality_reduction", "mortality_ci_lo", "mortality_ci_hi", "Mortality Reduction", ACCENT),
]

for ax, (col, lo, hi, label, color) in zip(axes, metrics):
    ax.set_facecolor(SURFACE)
    vals = df_meta[col].values
    los = df_meta[lo].values
    his = df_meta[hi].values
    strats = [s.replace(" ", "\n") for s in df_meta["strategy"]]
    colors = [STRAT_COLORS[s] for s in df_meta["strategy"]]

    bars = ax.barh(strats, vals, xerr=[vals - los, his - vals],
                   color=colors, alpha=0.85, height=0.6,
                   error_kw={"ecolor": "white", "capsize": 4, "lw": 1.5})
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9, color=TEXT)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Rate (95% CI)", fontsize=9)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("Fig 2 — Comparative Outcomes Across AMR Alternative Strategies",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("figures/fig2_pooled_outcomes.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig2 ✓")

# ── Fig 3: Molecule Exception Map (scatter: novelty vs resistance-proof) ──────
fig, ax = plt.subplots(figsize=(13, 8), facecolor=BG)
ax.set_facecolor(SURFACE)

class_colors = {
    "Beta-lactam analog": "#ef4444", "Lipopeptide": "#f97316",
    "Glycopeptide analog": "#eab308", "Antimicrobial Peptide": "#22c55e",
    "CRISPR-guided": "#06b6d4", "Phage-derived endolysin": "#3b82f6",
    "Siderophore-antibiotic": "#8b5cf6", "AI-designed scaffold": "#ec4899",
    "Bacteriocin": "#14b8a6", "Quorum sensing inhibitor": "#f59e0b",
}

for _, row in df_mol.iterrows():
    color = class_colors.get(row["class"], "#94a3b8")
    size = row["activity_score"] * 300
    ax.scatter(row["novelty_score"], row["resistance_proof_score"],
               s=size, color=color, alpha=0.8, edgecolors="white", linewidths=0.8)
    if row["exception_score"] > 0.75:
        ax.annotate(row["name"], (row["novelty_score"], row["resistance_proof_score"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=TEXT)

# Exception zone
rect = mpatches.FancyBboxPatch((0.72, 0.78), 0.26, 0.20,
                                boxstyle="round,pad=0.01",
                                linewidth=2, edgecolor=PINK, facecolor="none",
                                linestyle="--", alpha=0.8)
ax.add_patch(rect)
ax.text(0.85, 1.00, "Exception Zone", ha="center", fontsize=9,
        color=PINK, fontweight="bold")

ax.axhline(0.78, color=MUTED, lw=0.8, ls=":", alpha=0.5)
ax.axvline(0.72, color=MUTED, lw=0.8, ls=":", alpha=0.5)
ax.set_xlabel("Novelty Score", fontsize=11)
ax.set_ylabel("Resistance-Proof Score", fontsize=11)
ax.set_title("Fig 3 — Molecule Exception Map\n"
             "Bubble size = antimicrobial activity · Exception Zone = top-right quadrant",
             fontsize=12, fontweight="bold")

legend_patches = [mpatches.Patch(color=c, label=k) for k, c in class_colors.items()]
ax.legend(handles=legend_patches, loc="lower left", fontsize=8,
          framealpha=0.3, facecolor=SURFACE, edgecolor=MUTED, ncol=2)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("figures/fig3_molecule_exception_map.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig3 ✓")

# ── Fig 4: ESKAPE Pathogen Heatmap — Efficacy by strategy ────────────────────
fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
ax.set_facecolor(BG)

pivot = df_sub.pivot(index="pathogen", columns="strategy", values="pooled_efficacy")
pivot = pivot.reindex(columns=strategies)

pathogens_list = pivot.index.tolist()
strats_list = pivot.columns.tolist()

import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list(
    "amr", ["#1e293b", "#0369a1", "#22c55e", "#f97316"], N=256
)

im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0.4, vmax=1.0)
ax.set_xticks(range(len(strats_list)))
ax.set_xticklabels([s.replace(" ", "\n") for s in strats_list], fontsize=9)
ax.set_yticks(range(len(pathogens_list)))
ax.set_yticklabels(pathogens_list, fontsize=10)

for i in range(len(pathogens_list)):
    for j in range(len(strats_list)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val < 0.75 else "#0f172a")

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Pooled Efficacy", color=TEXT, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

ax.set_title("Fig 4 — Efficacy Heatmap: Strategy × ESKAPE Pathogen\n"
             "(Subgroup meta-analysis, random-effects pooled rates)",
             fontsize=12, fontweight="bold", pad=15)
fig.tight_layout()
fig.savefig("figures/fig4_eskape_heatmap.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig4 ✓")

# ── Fig 5: Top 10 Exception Molecules — Radar + Bar ──────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

# Left: Top 10 bar chart
top10 = df_mol.head(10)
colors_top = [class_colors.get(c, "#94a3b8") for c in top10["class"]]
bars = ax1.barh(range(10), top10["exception_score"], color=colors_top,
                alpha=0.85, height=0.7)
ax1.set_yticks(range(10))
ax1.set_yticklabels(top10["name"], fontsize=10)
ax1.set_facecolor(SURFACE)
ax1.set_xlabel("Exception Score", fontsize=11)
ax1.set_title("Top 10 Exception Molecules", fontsize=12, fontweight="bold")
for bar, (_, row) in zip(bars, top10.iterrows()):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.3f}", va="center", fontsize=9, color=TEXT)
ax1.axvline(0.75, color=YELLOW, lw=1.5, ls="--", alpha=0.7)
ax1.text(0.75, 9.7, " Exception\n threshold", fontsize=8, color=YELLOW)
ax1.set_xlim(0, 1.05)
ax1.grid(axis="x", alpha=0.3)

# Right: Spider/radar for top molecule
ax2.set_facecolor(SURFACE)
top1 = df_mol.iloc[0]
categories = ["Activity", "Resist-Proof", "Novelty", "Low Toxicity", "Exception"]
values = [
    top1["activity_score"],
    top1["resistance_proof_score"],
    top1["novelty_score"],
    1 - top1["toxicity_index"],
    top1["exception_score"],
]

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
values_plot = values + values[:1]

# Recreate as polar
fig.delaxes(ax2)
ax2 = fig.add_subplot(1, 2, 2, polar=True, facecolor=SURFACE)
ax2.set_facecolor(SURFACE)
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)

ax2.plot(angles, values_plot, color=PINK, lw=2.5)
ax2.fill(angles, values_plot, color=PINK, alpha=0.25)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=10, color=TEXT)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
ax2.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7, color=MUTED)
ax2.grid(color=MUTED, alpha=0.3)
ax2.set_title(f"Profile: {top1['name']}\n({top1['class']})",
              fontsize=11, fontweight="bold", pad=20)

fig.suptitle("Fig 5 — Exception Molecules: Ranking & Profile",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("figures/fig5_exception_molecules.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig5 ✓")

# ── Fig 6: Evidence Synthesis — I² heterogeneity + funnel ────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

# Left: I² by strategy and outcome
outcomes = ["Efficacy", "Resistance\nEmergence", "Mortality\nReduction"]
i2_data = np.array([
    df_meta["efficacy_i2"].values,
    df_meta["resistance_i2"].values,
    df_meta["mortality_i2"].values,
]).T

x = np.arange(len(outcomes))
width = 0.15
ax1.set_facecolor(SURFACE)

for i, (strat, i2_vals) in enumerate(zip(df_meta["strategy"], i2_data)):
    color = STRAT_COLORS[strat]
    offset = (i - 2) * width
    bars = ax1.bar(x + offset, i2_vals, width, label=strat.replace(" ", "\n"),
                   color=color, alpha=0.85)

ax1.axhline(75, color=RED, lw=1.5, ls="--", alpha=0.7)
ax1.axhline(50, color=YELLOW, lw=1.2, ls=":", alpha=0.6)
ax1.text(2.45, 76, "High I²", fontsize=8, color=RED)
ax1.text(2.45, 51, "Moderate I²", fontsize=8, color=YELLOW)
ax1.set_xticks(x)
ax1.set_xticklabels(outcomes, fontsize=10)
ax1.set_ylabel("I² (%)", fontsize=11)
ax1.set_ylim(0, 115)
ax1.set_title("Heterogeneity (I²) by Strategy & Outcome", fontsize=11, fontweight="bold")
ax1.legend(fontsize=7, framealpha=0.3, facecolor=SURFACE, edgecolor=MUTED,
           loc="upper left", ncol=2)
ax1.grid(axis="y", alpha=0.3)

# Right: Funnel plot (efficacy)
ax2.set_facecolor(SURFACE)
all_rates = df_studies["efficacy_rate"].values
all_ns = df_studies["n_patients"].values.astype(float)
se = np.sqrt(1 / (all_ns * all_rates * (1 - all_rates)))
logits = np.log(all_rates / (1 - all_rates))

colors_f = [STRAT_COLORS[s] for s in df_studies["strategy"]]
ax2.scatter(logits, 1 / se, c=colors_f, alpha=0.7, s=40, edgecolors="white", lw=0.5)

pooled_logit = np.mean(logits)
max_se = max(1 / se) * 1.1
se_range = np.linspace(0, max(1 / se), 100)
ax2.plot([pooled_logit - 1.96 * (1 / se_range[1:]),
          pooled_logit + 1.96 * (1 / se_range[1:])],
         [se_range[1:], se_range[1:]], color=MUTED, lw=0.5, alpha=0.3)

ax2.fill_betweenx(se_range[1:],
                  pooled_logit - 1.96 * (1 / se_range[1:]),
                  pooled_logit + 1.96 * (1 / se_range[1:]),
                  color=MUTED, alpha=0.1)
ax2.axvline(pooled_logit, color=ACCENT, lw=1.5, ls="--")
ax2.set_xlabel("Logit Efficacy Rate", fontsize=11)
ax2.set_ylabel("Precision (1/SE)", fontsize=11)
ax2.set_title("Funnel Plot — Publication Bias Assessment", fontsize=11, fontweight="bold")
ax2.grid(alpha=0.3)

patches = [mpatches.Patch(color=c, label=s.replace(" ", "\n"))
           for s, c in STRAT_COLORS.items()]
ax2.legend(handles=patches, fontsize=7, framealpha=0.3,
           facecolor=SURFACE, edgecolor=MUTED, loc="lower right")

fig.suptitle("Fig 6 — Heterogeneity & Publication Bias Analysis",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("figures/fig6_heterogeneity_funnel.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("fig6 ✓")

print("\n✅ All figures -> figures/")
