"""Matplotlib figures (static PNGs) with a consistent, colour-blind-aware style.

Colours follow a validated categorical order (blue, orange, aqua); identity
is never colour-only: every multi-series figure has a legend or labels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#9a9893"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "text.color": TEXT,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def _season_label(si: int) -> str:
    return f"{'Pre' if si % 2 == 0 else 'Post'}-{si // 2}"


def coverage_heatmap(long: pd.DataFrame, path):
    codes = {"reading": 0, "closed": 1, "dead": 2, "blank": 3, "not_listed": 4}
    names = ["Reading", "Closed", "Dead", "Blank", "Not listed in file"]
    colors = [SERIES[0], SERIES[1], "#7a5a00", "#d9d8d4", "#f3f2ef"]
    piv = long.assign(code=long["status"].map(codes)).pivot(index="well_id", columns="season_index", values="code")
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(piv.to_numpy(), aspect="auto", cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5,
              interpolation="nearest")
    ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
    ax.set_xticks(range(len(piv.columns)), [_season_label(c) for c in piv.columns], rotation=90, fontsize=7)
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5)
    ax.set_title("Reading availability per Lahore monitoring location and season", pad=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def well_trends(long: pd.DataFrame, well_ids, path, title="Depth to water table, selected wells"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, wid in enumerate(well_ids[:3]):
        g = long[(long["well_id"] == wid)].sort_values("season_index")
        x = g["season_index"] / 2.0
        ax.plot(x, g["depth_ft"], marker="o", ms=4, color=SERIES[i], label=wid)
    ax.invert_yaxis()
    ax.set_xlabel("Year (pre-monsoon = .0, post-monsoon = .5)")
    ax.set_ylabel("Depth to water below ground (ft)\n(down = deeper)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def median_depth_by_season(long: pd.DataFrame, path):
    ok = long[long["status"] == "reading"]
    agg = ok.groupby("season_index")["depth_ft"].agg(["median", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (season, lab) in enumerate([(0, "Pre-monsoon"), (1, "Post-monsoon")]):
        g = agg[agg["season_index"] % 2 == season]
        ax.plot(g["season_index"] // 2, g["median"], marker="o", ms=5, color=SERIES[i], label=lab)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel("Median depth to water (ft)\n(down = deeper)")
    ax.set_title("Median depth across wells with a reading (the set of wells changes over time)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def model_comparison(metrics: pd.DataFrame, path):
    m = metrics[metrics["split"] == "test"].sort_values("mae_ft", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(m) + 1.5))
    y = np.arange(len(m))
    ax.barh(y, m["mae_ft"], color=SERIES[0], height=0.55)
    for yi, v, r in zip(y, m["mae_ft"], m["rmse_ft"]):
        ax.text(v + 0.05, yi, f"MAE {v:.2f} ft  |  RMSE {r:.2f} ft", va="center", fontsize=9, color=TEXT2)
    ax.set_yticks(y, m["model"])
    ax.set_xlabel("Mean absolute error on held-out test period (ft)")
    ax.set_title("Next-season depth forecast error (lower is better)")
    ax.set_xlim(0, m["mae_ft"].max() * 1.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predicted_vs_actual(test: pd.DataFrame, preds: dict, path):
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo = min(test["target_depth_ft"].min(), min(p.min() for p in preds.values())) - 2
    hi = max(test["target_depth_ft"].max(), max(p.max() for p in preds.values())) + 2
    for ax, (i, (name, p)) in zip(axes, enumerate(preds.items())):
        ax.plot([lo, hi], [lo, hi], color=NEUTRAL, lw=1, ls="--", label="perfect forecast")
        ax.scatter(test["target_depth_ft"], p, s=28, color=SERIES[i], edgecolor=SURFACE, linewidth=1, label=name)
        ax.set_title(name)
        ax.set_xlabel("Observed depth (ft)")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Forecast depth (ft)")
    fig.suptitle("Held-out test period: forecast vs observed", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def change_errors(test: pd.DataFrame, preds: dict, path):
    """Forecast vs observed *change* — harder and more honest than levels."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    obs = test["target_change_ft"]
    lim = max(abs(obs).max(), 1) + 1
    ax.plot([-lim, lim], [-lim, lim], color=NEUTRAL, lw=1, ls="--", label="perfect forecast")
    for i, (name, p) in enumerate(list(preds.items())[:3]):
        ax.scatter(obs, p - test["depth_t_ft"], s=26, color=SERIES[i], edgecolor=SURFACE, linewidth=1,
                   label=name, alpha=0.9)
    ax.set_xlabel("Observed seasonal change (ft, + = water deeper)")
    ax.set_ylabel("Forecast seasonal change (ft)")
    ax.set_title("Seasonal change: forecast vs observed (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_importance(imp: pd.DataFrame, path, title):
    imp = imp.sort_values("mae_increase_ft")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(imp) + 1.5))
    y = np.arange(len(imp))
    ax.barh(y, imp["mae_increase_ft"], xerr=imp["std_ft"], color=SERIES[0], height=0.55,
            error_kw={"ecolor": TEXT2, "lw": 1, "capsize": 3})
    ax.axvline(0, color=TEXT2, lw=1)
    ax.set_yticks(y, imp["feature_group"])
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (ft)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(histories: list, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, h in enumerate(histories[:1]):
        ax.plot(h["loss"], color=SERIES[0], label="training loss")
        if "val_loss" in h:
            ax.plot(h["val_loss"], color=SERIES[1], label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber loss (scaled target)")
    ax.set_title("Neural network training (first ensemble member)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
