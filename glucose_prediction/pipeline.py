"""End-to-end dataset assembly, training and evaluation."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .data import STEP_MINUTES, load_patient, parse_ohio_xml
from .features import Windows, add_features, make_windows
from .metrics import horizon_report
from .model import (GlucoseForecaster, ModelConfig, RidgeForecaster, linear_extrapolation_forecast,
                    persistence_forecast)

REPORT_HORIZONS = (6, 12)  # 30 and 60 minutes, the standard OhioT1DM challenge horizons


def group_by_patient(paths) -> dict[str, list]:
    """Map patient id -> XML paths ordered by their first CGM timestamp."""
    groups = defaultdict(list)
    for p in paths:
        rec = parse_ohio_xml(p)
        start = rec.glucose.index.min() if len(rec.glucose) else pd.Timestamp.max
        groups[rec.patient_id].append((start, p))
    return {pid: [p for _, p in sorted(items)] for pid, items in groups.items()}


def chronological_split(w: Windows, val_fraction: float, horizon: int) -> tuple[Windows, Windows]:
    """Split one patient's windows in time: the last ``val_fraction`` becomes validation.

    Training samples whose targets would reach into the validation period are dropped
    (an embargo of ``horizon`` steps), so no validation reading is ever a training target.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    order = np.argsort(w.times.to_numpy())
    w = w.subset(order)
    cut = w.times[int(len(w) * (1 - val_fraction))]
    embargo = pd.Timedelta(minutes=horizon * STEP_MINUTES)
    train = w.subset(np.asarray(w.times <= cut - embargo))
    val = w.subset(np.asarray(w.times >= cut))
    return train, val


def build_training_sets(train_paths, config: ModelConfig, val_fraction: float = 0.2):
    """Windows for every patient in ``train_paths`` split chronologically per patient."""
    trains, vals = [], []
    for pid, paths in group_by_patient(train_paths).items():
        feats = add_features(load_patient(paths))
        w = make_windows(feats, config.history, config.horizon)
        tr, va = chronological_split(w, val_fraction, config.horizon)
        print(f"patient {pid}: {len(tr)} train / {len(va)} validation windows")
        trains.append(tr)
        vals.append(va)
    return Windows.concat(trains), Windows.concat(vals)


def build_test_windows(test_paths, config: ModelConfig, context_paths=()) -> dict[str, Windows]:
    """Windows anchored inside the test files, one entry per patient.

    Files in ``context_paths`` that belong to the same patient (typically the training XML,
    which ends right where the test XML begins) are prepended so that insulin/carbs on board and
    the input window are complete for the first test predictions. Only anchors that fall inside
    the test files are returned.
    """
    context = group_by_patient(context_paths) if context_paths else {}
    out = {}
    for pid, paths in group_by_patient(test_paths).items():
        test_start = min(parse_ohio_xml(p).glucose.index.min() for p in paths)
        grid = load_patient(list(context.get(pid, [])) + list(paths))
        w = make_windows(add_features(grid), config.history, config.horizon)
        out[pid] = w.subset(np.asarray(w.times >= test_start.round("5min")))
    return out


def evaluate_models(models: dict, windows: Windows, horizon: int, horizons=REPORT_HORIZONS) -> dict:
    """Metrics per model and horizon. ``models`` maps name -> callable(Windows) -> predictions."""
    return {name: horizon_report(windows.y, fn(windows), horizons) for name, fn in models.items()}


def standard_model_suite(net: GlucoseForecaster | None, ridge: RidgeForecaster | None, horizon: int) -> dict:
    suite = {
        "persistence": lambda w: persistence_forecast(w, horizon),
        "linear_extrapolation": lambda w: linear_extrapolation_forecast(w, horizon),
    }
    if ridge is not None:
        suite["ridge_arx"] = ridge.predict
    if net is not None:
        suite["neural_net"] = net.predict
    return suite


def format_report(report: dict, horizons=REPORT_HORIZONS) -> str:
    """Render ``{model: {"30min": metrics}}`` as a fixed-width table."""
    lines = []
    header = f"{'model':<22}{'horizon':>8}{'n':>7}{'RMSE':>8}{'MAE':>8}{'MARD%':>8}{'EGA A%':>8}{'A+B%':>7}"
    header += f"{'hypo sens':>10}{'hypo prec':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, per_h in report.items():
        for h in horizons:
            m = per_h[f"{h * STEP_MINUTES}min"]
            if m.get("n", 0) == 0:
                continue
            hypo = m["hypo_alert"]
            lines.append(
                f"{name:<22}{h * STEP_MINUTES:>6}mn{m['n']:>7}{m['rmse']:>8.2f}{m['mae']:>8.2f}"
                f"{m['mard_pct']:>8.2f}{m['clarke_A_pct']:>8.1f}{m['clarke_A_pct'] + m['clarke_B_pct']:>7.1f}"
                f"{hypo['sensitivity']:>10.2f}{hypo['precision']:>10.2f}"
            )
    return "\n".join(lines)
