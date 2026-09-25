"""Feature engineering and causal windowing.

Every feature at time ``t`` is computed only from data at or before ``t`` so that the model
never sees the future. Unit tests in ``tests/test_features.py`` check this property.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import STEP_MINUTES

# Exponential insulin action model (as used by OpenAPS / Loop) for rapid-acting insulin.
INSULIN_DURATION_MIN = 360.0
INSULIN_PEAK_MIN = 75.0
# The same curve family is used as a simple carbohydrate absorption model.
CARB_DURATION_MIN = 240.0
CARB_PEAK_MIN = 60.0

# Interior CGM gaps up to this length are linearly interpolated in model *inputs*.
MAX_INTERP_STEPS = 6  # 30 minutes

FEATURE_COLUMNS = [
    "glucose",
    "glucose_roc",  # rate of change, mg/dL per 5 min
    "glucose_imputed",  # 1 if the input value was interpolated
    "iob",  # insulin on board (U)
    "insulin_activity",  # U absorbed per 5 min
    "cob",  # carbs on board (g)
    "carb_activity",  # g absorbed per 5 min
    "bolus_insulin",
    "carbs",
    "basal_rate",  # U/h
    "exercise",
    "tod_sin",
    "tod_cos",
]


def exponential_action_curves(duration_min: float, peak_min: float, step_min: float = STEP_MINUTES):
    """Return ``(activity, remaining)`` kernels of the exponential insulin action model.

    ``activity[k]`` is the fraction of a unit dose absorbed during step ``k`` after delivery and
    ``remaining[k]`` the fraction still on board at the end of step ``k`` (``remaining[0]`` ~ 1).
    Reference: https://github.com/LoopKit/Loop/issues/388
    """
    td, tp = float(duration_min), float(peak_min)
    if not 0 < tp < td / 2:
        raise ValueError("peak must lie in (0, duration/2)")
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    s = 1 / (1 - a + (1 + a) * np.exp(-td / tau))
    t = np.arange(0, td + step_min, step_min)
    remaining = 1 - s * (1 - a) * ((t**2 / (tau * td * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1)
    remaining = np.clip(remaining, 0.0, 1.0)
    remaining[-1] = 0.0
    activity = -np.diff(remaining, prepend=1.0)
    return activity, remaining


def _causal_convolve(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(x, kernel)[: len(x)]


def causal_interpolate(glucose: pd.Series, limit: int = MAX_INTERP_STEPS) -> pd.Series:
    """Linearly interpolate interior gaps of at most ``limit`` steps.

    Longer gaps stay ``NaN``. Values after the last observation are never filled, and
    :func:`make_windows` only builds a window when its last step is a real reading, so every
    interpolated input value lies between two readings that are both at or before the forecast
    time; no future information leaks into the inputs.
    """
    isna = glucose.isna()
    run_id = (~isna).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = glucose.interpolate(method="linear", limit_area="inside")
    return filled.where(~isna | (run_len <= limit))


def add_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from a 5-minute grid produced by :func:`data.build_timeseries`."""
    df = pd.DataFrame(index=grid.index)
    raw = grid["glucose"]
    glucose = causal_interpolate(raw)
    df["glucose"] = glucose
    df["glucose_roc"] = glucose.diff()
    df["glucose_imputed"] = (raw.isna() & glucose.notna()).astype(float)

    insulin = (grid["basal_insulin"] + grid["bolus_insulin"]).to_numpy()
    ins_act, ins_rem = exponential_action_curves(INSULIN_DURATION_MIN, INSULIN_PEAK_MIN)
    df["iob"] = _causal_convolve(insulin, ins_rem)
    df["insulin_activity"] = _causal_convolve(insulin, ins_act)

    carbs = grid["carbs"].to_numpy()
    carb_act, carb_rem = exponential_action_curves(CARB_DURATION_MIN, CARB_PEAK_MIN)
    df["cob"] = _causal_convolve(carbs, carb_rem)
    df["carb_activity"] = _causal_convolve(carbs, carb_act)

    df["bolus_insulin"] = grid["bolus_insulin"]
    df["carbs"] = grid["carbs"]
    df["basal_rate"] = grid["basal_insulin"] * (60.0 / STEP_MINUTES)
    df["exercise"] = grid["exercise"]

    minutes = df.index.hour * 60 + df.index.minute
    angle = 2 * np.pi * minutes / (24 * 60)
    df["tod_sin"] = np.sin(angle)
    df["tod_cos"] = np.cos(angle)

    df["target"] = raw  # only real CGM readings are ever used as targets
    df.attrs.update(grid.attrs)
    return df[FEATURE_COLUMNS + ["target"]]


@dataclass
class Windows:
    """Supervised samples built from a feature frame.

    X          (n, history, n_features) model inputs
    y          (n, horizon) future glucose in mg/dL (``NaN`` where no CGM reading exists)
    last       (n,) glucose at the forecast time t (the anchor)
    times      (n,) forecast times t
    """

    X: np.ndarray
    y: np.ndarray
    last: np.ndarray
    times: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.last)

    def subset(self, mask) -> "Windows":
        return Windows(self.X[mask], self.y[mask], self.last[mask], self.times[mask])

    @staticmethod
    def concat(parts: list["Windows"]) -> "Windows":
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("no windows to concatenate")
        return Windows(
            np.concatenate([p.X for p in parts]),
            np.concatenate([p.y for p in parts]),
            np.concatenate([p.last for p in parts]),
            pd.DatetimeIndex(np.concatenate([p.times.to_numpy() for p in parts])),
        )


def make_windows(features: pd.DataFrame, history: int, horizon: int,
                 require_target: bool = True) -> Windows:
    """Slide a causal window over ``features``.

    A sample anchored at step ``t`` uses inputs ``t-history+1 .. t`` and targets ``t+1 .. t+horizon``.
    It is kept only if

    * the anchor ``t`` is a real (not interpolated) CGM reading,
    * no input step has missing glucose (gaps > 30 min break the sequence),
    * with ``require_target``, at least one target reading exists.
    """
    X_all = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target = features["target"].to_numpy(dtype=np.float32)
    n = len(features)
    if n < history:
        return Windows(np.empty((0, history, len(FEATURE_COLUMNS)), np.float32),
                       np.empty((0, horizon), np.float32), np.empty(0, np.float32),
                       pd.DatetimeIndex([]))

    X_all = np.nan_to_num(X_all, nan=0.0)  # roc of first step; glucose NaNs are filtered below
    glucose_ok = features["glucose"].notna().to_numpy()
    # Count of valid glucose values inside each trailing window of length `history`.
    csum = np.concatenate([[0], np.cumsum(glucose_ok)])
    anchors = np.arange(history - 1, n)
    valid = (csum[anchors + 1] - csum[anchors + 1 - history]) == history
    valid &= ~np.isnan(target[anchors])

    y = np.full((len(anchors), horizon), np.nan, dtype=np.float32)
    for h in range(1, horizon + 1):
        idx = anchors + h
        ok = idx < n
        y[ok, h - 1] = target[idx[ok]]
    if require_target:
        valid &= ~np.all(np.isnan(y), axis=1)

    anchors, y = anchors[valid], y[valid]
    offsets = np.arange(-history + 1, 1)
    X = X_all[anchors[:, None] + offsets[None, :]]
    return Windows(X, y, target[anchors], features.index[anchors])
