"""Detection metrics. EER (Equal Error Rate) is the standard metric in the
ASVspoof literature — the operating point where the false-accept rate and
false-reject rate are equal; lower is better. 0% = perfect separation,
50% = no better than chance."""
import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels, scores) -> tuple:
    """labels: 1 = bonafide, 0 = spoof (array-like).
    scores: higher = more likely bonafide, e.g. sigmoid(logit) from Detector.
    Returns (eer, threshold) as floats, eer in [0, 1]."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])
