"""Accuracy and clinical metrics for glucose forecasts (all in mg/dL)."""
from __future__ import annotations

import numpy as np

HYPO_THRESHOLD = 70.0
HYPER_THRESHOLD = 180.0


def clarke_zones(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Clarke error grid zone ('A'..'E') for each (reference, prediction) pair.

    Zone boundaries follow Clarke et al., Diabetes Care 10(5):622-628, 1987.
    """
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    zones = np.full(ref.shape, "B", dtype="<U1")

    d = (((ref >= 240) & (pred >= 70) & (pred <= 180))
         | ((ref <= 175 / 3) & (pred >= 70) & (pred <= 180))
         | ((ref >= 175 / 3) & (ref <= 70) & (pred >= 1.2 * ref)))
    c = (((ref >= 70) & (ref <= 290) & (pred >= ref + 110))
         | ((ref >= 130) & (ref <= 180) & (pred <= 1.4 * ref - 182)))
    e = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    a = ((ref <= 70) & (pred <= 70)) | ((pred >= 0.8 * ref) & (pred <= 1.2 * ref))

    # Assign in reverse priority so that A > E > C > D > B wins.
    zones[d] = "D"
    zones[c] = "C"
    zones[e] = "E"
    zones[a] = "A"
    return zones


def threshold_alert_scores(ref: np.ndarray, pred: np.ndarray, threshold: float, below: bool = True) -> dict:
    """Sample-wise sensitivity / precision of an alert raised when the forecast crosses ``threshold``."""
    actual = ref < threshold if below else ref > threshold
    alarm = pred < threshold if below else pred > threshold
    tp = int(np.sum(actual & alarm))
    fp = int(np.sum(~actual & alarm))
    fn = int(np.sum(actual & ~alarm))
    return {
        "events": int(actual.sum()),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }


def regression_metrics(ref, pred) -> dict:
    """RMSE, MAE, MARD, Clarke zone percentages and hypo/hyper alert scores."""
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = ~(np.isnan(ref) | np.isnan(pred))
    ref, pred = ref[ok], pred[ok]
    if len(ref) == 0:
        return {"n": 0}
    err = pred - ref
    zones = clarke_zones(ref, pred)
    out = {
        "n": int(len(ref)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mard_pct": float(100 * np.mean(np.abs(err) / ref)),
        "bias": float(np.mean(err)),
    }
    for z in "ABCDE":
        out[f"clarke_{z}_pct"] = float(100 * np.mean(zones == z))
    out["hypo_alert"] = threshold_alert_scores(ref, pred, HYPO_THRESHOLD, below=True)
    out["hyper_alert"] = threshold_alert_scores(ref, pred, HYPER_THRESHOLD, below=False)
    return out


def horizon_report(y_true: np.ndarray, y_pred: np.ndarray, horizons_steps, step_minutes: int = 5) -> dict:
    """Metrics per forecast horizon, keyed like ``"30min"``."""
    return {
        f"{h * step_minutes}min": regression_metrics(y_true[:, h - 1], y_pred[:, h - 1])
        for h in horizons_steps
    }
