import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

_MODEL_COLOR = "tab:blue"
_BASELINE_COLOR = "gray"
_POINT_COLOR = "tab:red"
_PAD = 0.03  # extra room so curves lying on the borders (e.g. AUC = 1.0) stay visible


def _style_axis(ax, xlabel, ylabel, title):
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlim=(-_PAD, 1 + _PAD),
        ylim=(-_PAD, 1 + _PAD),
        aspect="equal",
    )
    for spine in ax.spines.values():
        spine.set_color("lightgray")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right" if title.startswith("ROC") else "lower left", fontsize=9)


def plot_roc_pr_curves(y, y_hat, figsize=(13, 6), save_path=None, title=None):
    y = np.ravel(y)
    y_hat = np.ravel(y_hat)

    roc_auc = roc_auc_score(y, y_hat)
    pr_auc = average_precision_score(y, y_hat)
    prevalence = float(np.mean(y))

    fpr, tpr, roc_thresholds = roc_curve(y, y_hat)
    precision, recall, pr_thresholds = precision_recall_curve(y, y_hat)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ---------------- ROC ----------------
    ax = axes[0]
    ax.fill_between(fpr, tpr, alpha=0.15, color=_MODEL_COLOR)
    ax.plot(fpr, tpr, color=_MODEL_COLOR, lw=2.5,
            label=f"Modelo (AUC-ROC = {roc_auc:.4f})", zorder=3, clip_on=False)
    ax.plot([0, 1], [0, 1], "--", color=_BASELINE_COLOR, lw=1.5, label="Acaso (AUC-ROC = 0.50)")

    # Best threshold by Youden's J = TPR - FPR (closest point to the top-left corner)
    best = int(np.argmax(tpr - fpr))
    ax.scatter(fpr[best], tpr[best], s=80, color=_POINT_COLOR, zorder=4, clip_on=False,
               label=f"Melhor limiar: TPR = {tpr[best]:.2f}, FPR = {fpr[best]:.2f}")
    _style_axis(ax, "Taxa de falsos positivos (FPR)", "Taxa de verdadeiros positivos (TPR / Recall)", "ROC Curve")

    # ---------------- Precision-Recall ----------------
    ax = axes[1]
    ax.fill_between(recall, precision, step="post", alpha=0.15, color=_MODEL_COLOR)
    ax.plot(recall, precision, color=_MODEL_COLOR, lw=2.5, drawstyle="steps-post",
            label=f"Modelo (AP = {pr_auc:.4f})", zorder=3, clip_on=False)
    ax.axhline(prevalence, linestyle="--", color=_BASELINE_COLOR, lw=1.5,
               label=f"Acaso (precisão = % de anomalias = {prevalence:.2f})")

    # Best threshold by F1 (last precision/recall pair has no threshold, so it is skipped)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = int(np.argmax(f1))
    ax.scatter(r[best], p[best], s=80, color=_POINT_COLOR, zorder=4, clip_on=False,
               label=f"Melhor F1 = {f1[best]:.2f} (P = {p[best]:.2f}, R = {r[best]:.2f})")
    _style_axis(ax, "Recall (anomalias detectadas)", "Precisão (acusações corretas)", "Precision-Recall Curve")

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return roc_auc, pr_auc
