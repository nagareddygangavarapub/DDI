"""
run_visualization.py — Generate EDA charts for DrugSafe AI datasets.

Produces 3 figure files:
    outputs/fig1_fda_overview.png      — FDA label dataset overview (4 charts)
    outputs/fig2_ddi_pairs.png         — DDI pairs dataset analysis (4 charts)
    outputs/fig3_nlp_analysis.png      — NLP / text analysis (4 charts)

Usage:
    cd ddi_rag
    python run_visualization.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualization import (
    setup_plot_style, styled_barh,
    get_ngrams_sklearn, get_tfidf_top_terms,
    COLORS, PALETTE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent
ROOT      = BASE.parent                                      # project root
FDA_CSV   = ROOT / "data" / "datasets" / "clean_ddi_dataset.csv"
DDI_CSV   = ROOT / "data" / "datasets" / "fully_processed_dataset.csv"
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

setup_plot_style()

print("Loading datasets...")
fda = pd.read_csv(FDA_CSV)
ddi = pd.read_csv(DDI_CSV)

# Drop junk rows from DDI pairs
ddi = ddi[
    (ddi["Drug 1"] != "Unknown_Drug") &
    (ddi["Interaction Description"] != "No_description")
].reset_index(drop=True)

print(f"  FDA labels : {fda.shape[0]:,} rows × {fda.shape[1]} cols")
print(f"  DDI pairs  : {ddi.shape[0]:,} rows × {ddi.shape[1]} cols")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — FDA Label Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating Figure 1 — FDA Label Overview...")
fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle("DrugSafe AI — FDA Label Dataset Overview", fontsize=16, fontweight="bold", y=1.01)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

# 1A — Top 15 Product Types
ax = axes[0, 0]
prod_counts = fda["openfda_product_type"].value_counts().head(15)
if not prod_counts.empty:
    styled_barh(ax, prod_counts.index.tolist(), prod_counts.values.tolist(),
                COLORS["accent1"], "Top Product Types", "Count")
else:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Top Product Types")

# 1B — Top 15 Routes of Administration
ax = axes[0, 1]
route_counts = fda["openfda_route"].value_counts().head(15)
if not route_counts.empty:
    styled_barh(ax, route_counts.index.tolist(), route_counts.values.tolist(),
                COLORS["accent2"], "Top Routes of Administration", "Count")
else:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Routes of Administration")

# 1C — Column completeness (% non-null)
ax = axes[1, 0]
text_cols = ["drug_interactions", "warnings", "adverse_reactions",
             "contraindications", "clinical_pharmacology"]
completeness = [(fda[c].notna() & (fda[c].astype(str).str.strip() != "")).mean() * 100
                for c in text_cols]
col_labels = [c.replace("_", "\n") for c in text_cols]
bars = ax.bar(col_labels, completeness, color=PALETTE[:len(text_cols)], alpha=0.85, width=0.6)
ax.set_title("Column Completeness (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("% Non-empty", fontsize=9)
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, completeness):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
            f"{val:.1f}%", ha="center", fontsize=9, color=COLORS["sub"])

# 1D — Text length distribution for drug_interactions
ax = axes[1, 1]
lengths = fda["drug_interactions"].dropna().astype(str).str.split().apply(len)
ax.hist(lengths, bins=40, color=COLORS["accent3"], alpha=0.8, edgecolor="white")
ax.set_title("Drug Interactions — Word Count Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Word Count per Entry", fontsize=9)
ax.set_ylabel("Number of Drugs", fontsize=9)
ax.axvline(lengths.median(), color=COLORS["accent2"], linestyle="--", linewidth=1.5,
           label=f"Median: {lengths.median():.0f} words")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

fig1.tight_layout()
out1 = OUT_DIR / "fig1_fda_overview.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"  Saved → {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — DDI Pairs Dataset Analysis
# ══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 2 — DDI Pairs Analysis...")
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("DrugSafe AI — DDI Pairs Dataset Analysis (187,950 interactions)",
              fontsize=16, fontweight="bold", y=1.01)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

# 2A — Top 15 Drug 1 (most interactions)
ax = axes[0, 0]
top_drug1 = ddi["Drug 1"].value_counts().head(15)
styled_barh(ax, top_drug1.index.tolist(), top_drug1.values.tolist(),
            COLORS["accent1"], "Top 15 Drugs by Interaction Count (Drug 1)", "Interactions")

# 2B — Top 15 Drug 2 (most targeted)
ax = axes[0, 1]
top_drug2 = ddi["Drug 2"].value_counts().head(15)
styled_barh(ax, top_drug2.index.tolist(), top_drug2.values.tolist(),
            COLORS["accent4"], "Top 15 Most Targeted Drugs (Drug 2)", "Times Targeted")

# 2C — Interaction description length distribution
ax = axes[1, 0]
desc_lens = ddi["Interaction Description"].astype(str).str.split().apply(len)
ax.hist(desc_lens, bins=30, color=COLORS["accent5"], alpha=0.85, edgecolor="white")
ax.set_title("Interaction Description — Word Count", fontsize=12, fontweight="bold")
ax.set_xlabel("Word Count", fontsize=9)
ax.set_ylabel("Number of Pairs", fontsize=9)
ax.axvline(desc_lens.mean(), color=COLORS["accent2"], linestyle="--", linewidth=1.5,
           label=f"Mean: {desc_lens.mean():.1f} words")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# 2D — Most common interaction types (keyword extraction from description)
ax = axes[1, 1]
interaction_keywords = {
    "increase": ddi["Interaction Description"].str.contains("increase", case=False, na=False).sum(),
    "decrease": ddi["Interaction Description"].str.contains("decrease", case=False, na=False).sum(),
    "serum concentration": ddi["Interaction Description"].str.contains("serum concentration", case=False, na=False).sum(),
    "metabolism": ddi["Interaction Description"].str.contains("metabolism", case=False, na=False).sum(),
    "toxic": ddi["Interaction Description"].str.contains("toxic", case=False, na=False).sum(),
    "risk": ddi["Interaction Description"].str.contains("risk", case=False, na=False).sum(),
    "excretion": ddi["Interaction Description"].str.contains("excretion", case=False, na=False).sum(),
    "absorption": ddi["Interaction Description"].str.contains("absorption", case=False, na=False).sum(),
    "bleeding": ddi["Interaction Description"].str.contains("bleeding", case=False, na=False).sum(),
    "hypertensive": ddi["Interaction Description"].str.contains("hypertensive", case=False, na=False).sum(),
}
kw_labels = list(interaction_keywords.keys())
kw_values = list(interaction_keywords.values())
sorted_pairs = sorted(zip(kw_values, kw_labels), reverse=True)
kw_values, kw_labels = [v for v, _ in sorted_pairs], [l for _, l in sorted_pairs]
styled_barh(ax, kw_labels, kw_values, COLORS["accent3"],
            "Interaction Type Keywords", "Occurrences")

fig2.tight_layout()
out2 = OUT_DIR / "fig2_ddi_pairs.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"  Saved → {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — NLP Analysis
# ══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 3 — NLP / Text Analysis...")
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle("DrugSafe AI — NLP Text Analysis", fontsize=16, fontweight="bold", y=1.01)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

# 3A — Top bigrams from drug_interactions (FDA labels)
ax = axes[0, 0]
labels_bi, counts_bi = get_ngrams_sklearn(
    fda["drug_interactions"].dropna(), ngram_range=(2, 2), top_k=15
)
if labels_bi:
    styled_barh(ax, labels_bi[::-1], counts_bi[::-1], COLORS["accent1"],
                "Top Bigrams — FDA Drug Interactions", "Count")
else:
    ax.set_title("Top Bigrams (no data)")

# 3B — TF-IDF top terms from warnings (FDA labels)
ax = axes[0, 1]
tfidf_terms, tfidf_scores = get_tfidf_top_terms(
    fda["warnings"].dropna(), top_k=15, ngram_range=(1, 2)
)
if tfidf_terms:
    styled_barh(ax, tfidf_terms, tfidf_scores, COLORS["accent2"],
                "TF-IDF Top Terms — FDA Warnings", "Mean TF-IDF Score")
else:
    ax.set_title("TF-IDF Terms (no data)")

# 3C — Top bigrams from DDI pair descriptions
ax = axes[1, 0]
ddi_labels_bi, ddi_counts_bi = get_ngrams_sklearn(
    ddi["Cleaned_Description"], ngram_range=(2, 2), top_k=15
)
if ddi_labels_bi:
    styled_barh(ax, ddi_labels_bi[::-1], ddi_counts_bi[::-1], COLORS["accent4"],
                "Top Bigrams — DDI Pair Descriptions", "Count")
else:
    ax.set_title("DDI Bigrams (no data)")

# 3D — Dataset size comparison (stacked bar)
ax = axes[1, 1]
categories = ["FDA\nLabel Vectors", "DDI Pair\nVectors", "Total ChromaDB\nVectors"]
values     = [558046, 187671, 745717]
colors_bar = [COLORS["accent1"], COLORS["accent3"], COLORS["accent5"]]
bars = ax.bar(categories, values, color=colors_bar, alpha=0.85, width=0.5)
ax.set_title("ChromaDB Index Composition", fontsize=12, fontweight="bold")
ax.set_ylabel("Vector Count", fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 8000,
            f"{val:,}", ha="center", fontsize=10, fontweight="bold", color=COLORS["sub"])

fig3.tight_layout()
out3 = OUT_DIR / "fig3_nlp_analysis.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print(f"  Saved → {out3}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"  All 3 figures saved to:  {OUT_DIR}")
print("=" * 50)
print(f"  fig1_fda_overview.png   — FDA dataset overview")
print(f"  fig2_ddi_pairs.png      — DDI pairs analysis")
print(f"  fig3_nlp_analysis.png   — NLP / text analysis")
print("=" * 50)
