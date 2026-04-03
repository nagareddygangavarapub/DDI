"""
visualization.py — Plotting utilities for the DDI EDA notebook.

Import this module to apply consistent chart styling and use the
shared NLP helpers (clean_text, get_ngrams_sklearn, styled_barh).
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":      "white",
    "card":    "white",
    "accent1": "#58A6FF",
    "accent2": "#F78166",
    "accent3": "#3FB950",
    "accent4": "#D2A8FF",
    "accent5": "#FFA657",
    "text":    "black",
    "sub":     "#666666",
}
PALETTE = [
    COLORS["accent1"], COLORS["accent2"], COLORS["accent3"],
    COLORS["accent4"], COLORS["accent5"], "#79C0FF", "#FF7B72", "#56D364",
]

# ── Stop-word list for NLP ────────────────────────────────────────────────────
STOP_WORDS = list({
    "the","and","for","are","was","with","this","that","have","has","from",
    "not","been","will","can","its","into","more","other","such","they",
    "also","than","when","some","may","use","used","using","drug","drugs",
    "patient","patients","treatment","dose","should","clinical","effects",
    "adverse","reactions","following","therapy","based","including","however",
    "these","their","which","been","after","during","upon","those","while",
    "due","per","well","both","each","only","most","all","any","your",
    "there","two","one","three","four","five","study","studies","cases",
    "case","reported","potential","risk","increased","significant","associated",
    "administration","administered","medications","medication","information",
    "data","product","products","table","below","above","see","section",
    "does","about","between","through","must","because","known","possible",
    "therefore","although","since","where","were","being","having","without",
    "within","before","after","time","times","number","level","levels",
    "effect","action","activity","indicate","indicates","cause","causes",
    "contain","contains","include","includes","similar","specific","certain",
})
STOP_SET = set(STOP_WORDS)


def setup_plot_style() -> None:
    """Apply consistent matplotlib styling for all charts."""
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#cccccc",
        "axes.labelcolor":    "black",
        "xtick.color":        "black",
        "ytick.color":        "black",
        "text.color":         "black",
        "grid.color":         "#e0e0e0",
        "grid.linestyle":     "--",
        "grid.alpha":         0.6,
        "font.family":        "DejaVu Sans",
        "font.size":          12,
        "font.weight":        "bold",
        "axes.titlesize":     15,
        "axes.titleweight":   "bold",
        "axes.labelsize":     13,
        "axes.labelweight":   "bold",
        "xtick.labelsize":    14,
        "ytick.labelsize":    14,
        "legend.fontsize":    14,
        "figure.titlesize":   18,
        "figure.titleweight": "bold",
    })


def clean_text(text: str) -> str:
    """
    Lowercase, strip punctuation/numbers, remove stop words and short tokens.
    Returns a space-joined string suitable for CountVectorizer / TfidfVectorizer.
    """
    if not isinstance(text, str):
        return ""
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", " ", text)
    text   = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_SET and len(w) > 3]
    return " ".join(tokens)


def styled_barh(
    ax, labels, values, color, title: str, xlabel: str = "Count"
) -> None:
    """Draw a clean horizontal bar chart on an existing Axes object."""
    bars  = ax.barh(labels, values, color=color, height=0.65, alpha=0.88)
    ax.set_title(
        title, color=COLORS["text"], fontsize=12, fontweight="bold", pad=10
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    max_v = max(values) if values else 1
    for bar, val in zip(bars, values):
        label = (
            f"{val:.4f}" if isinstance(val, float) and val < 1
            else f"{int(val):,}"
        )
        ax.text(
            val + max_v * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8, color=COLORS["sub"],
        )


def get_ngrams_sklearn(
    series: pd.Series,
    ngram_range: tuple = (2, 2),
    top_k: int = 20,
):
    """
    Return the top-k n-gram labels and their counts from a text Series.

    Returns:
        (labels: list[str], counts: list[float])
    """
    corpus = series.dropna().apply(clean_text).tolist()
    corpus = [t for t in corpus if t.strip()]
    if not corpus:
        return [], []
    vec     = CountVectorizer(
        ngram_range=ngram_range, max_features=5_000, stop_words=STOP_WORDS
    )
    X       = vec.fit_transform(corpus)
    counts  = np.asarray(X.sum(axis=0)).flatten()
    vocab   = vec.get_feature_names_out()
    top_idx = counts.argsort()[-top_k:][::-1]
    return vocab[top_idx].tolist(), counts[top_idx].tolist()


def get_tfidf_top_terms(
    series: pd.Series,
    top_k: int = 15,
    ngram_range: tuple = (1, 2),
):
    """
    Return the top-k TF-IDF-scored terms and their mean scores from a Series.

    Returns:
        (terms: list[str], scores: list[float])
    """
    corpus = series.dropna().apply(clean_text).tolist()
    corpus = [t for t in corpus if t.strip()]
    if len(corpus) < 3:
        return [], []
    tfidf   = TfidfVectorizer(
        max_features=500, ngram_range=ngram_range, stop_words=STOP_WORDS
    )
    matrix  = tfidf.fit_transform(corpus)
    scores  = np.asarray(matrix.mean(axis=0)).flatten()
    vocab   = tfidf.get_feature_names_out()
    top_idx = scores.argsort()[-top_k:][::-1]
    return vocab[top_idx][::-1].tolist(), scores[top_idx][::-1].tolist()
