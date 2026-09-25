"""Forecast glucose 5-60 minutes ahead with a trained model, raise hypo/hyper alerts and plot.

Example::

    python -m glucose_prediction.predict --model artifacts/559 \
        --input data/559-ws-testing.xml --history data/559-ws-training.xml \
        --output forecasts.csv --plot forecast.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import STEP_MINUTES, load_patient, parse_ohio_xml
from .features import add_features, make_windows
from .metrics import HYPER_THRESHOLD, HYPO_THRESHOLD
from .model import GlucoseForecaster, persistence_forecast
from .pipeline import REPORT_HORIZONS, evaluate_models, format_report

ALERT_HORIZON_STEPS = 6  # alerts look 30 minutes ahead


def forecast(model: GlucoseForecaster, input_paths, history_paths=()) -> pd.DataFrame:
    """One row per forecast time inside ``input_paths`` with the predicted trajectory.

    Columns: ``glucose`` (current reading), ``pred_<m>min`` for every horizon, ``std_<m>min``
    (spread across ensemble members, a rough uncertainty estimate), ``actual_<m>min`` where a
    reading exists, and the ``hypo_alert`` / ``hyper_alert`` flags.
    """
    cfg = model.config
    input_paths = list(input_paths)
    start = min(parse_ohio_xml(p).glucose.index.min() for p in input_paths).round("5min")
    feats = add_features(load_patient(list(history_paths) + input_paths))
    w = make_windows(feats, cfg.history, cfg.horizon, require_target=False)
    w = w.subset(np.asarray(w.times >= start))
    if not len(w):
        raise ValueError("no forecast times with a complete input window in the input file")

    members = model.predict_members(w)
    pred, spread = members.mean(axis=0), members.std(axis=0)
    df = pd.DataFrame({"glucose": w.last}, index=w.times)
    for h in range(1, cfg.horizon + 1):
        df[f"pred_{h * STEP_MINUTES}min"] = pred[:, h - 1]
    for h in range(1, cfg.horizon + 1):
        df[f"std_{h * STEP_MINUTES}min"] = spread[:, h - 1]
    for h in range(1, cfg.horizon + 1):
        df[f"actual_{h * STEP_MINUTES}min"] = w.y[:, h - 1]

    ahead = pred[:, :ALERT_HORIZON_STEPS]
    df["hypo_alert"] = (ahead < HYPO_THRESHOLD).any(axis=1)
    df["hyper_alert"] = (ahead > HYPER_THRESHOLD).any(axis=1)
    df.index.name = "timestamp"
    df.attrs["windows"] = w
    return df


def plot_forecast(df: pd.DataFrame, recorded: pd.Series, path=None, days: float = 2.0, horizon_min: int = 30):
    """Recorded CGM, the ``horizon_min``-ahead forecast track and the latest 60-min trajectory."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = df.index[-1]
    begin = end - pd.Timedelta(days=days)
    rec = recorded[(recorded.index >= begin)]
    track = df.loc[df.index >= begin, f"pred_{horizon_min}min"]
    track.index = track.index + pd.Timedelta(minutes=horizon_min)  # plot at the time it refers to

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhspan(0, HYPO_THRESHOLD, color="#7b3294", alpha=0.08, label=f"hypoglycaemia (< {HYPO_THRESHOLD:.0f})")
    ax.axhspan(HYPER_THRESHOLD, 400, color="#d7191c", alpha=0.06, label=f"hyperglycaemia (> {HYPER_THRESHOLD:.0f})")
    ax.plot(rec.index, rec.to_numpy(), color="#404040", lw=1.4, label="recorded CGM")
    ax.plot(track.index, track.to_numpy(), color="#2c7bb6", lw=1.1, alpha=0.9,
            label=f"forecast made {horizon_min} min earlier")

    last = df.iloc[-1]
    steps = [c for c in df.columns if c.startswith("pred_")]
    mins = np.array([int(c[5:-3]) for c in steps])
    traj = last[steps].to_numpy(dtype=float)
    sd = last[[c.replace("pred_", "std_") for c in steps]].to_numpy(dtype=float)
    t = end + pd.to_timedelta(mins, unit="min")
    ax.plot(np.r_[end, t], np.r_[last["glucose"], traj], color="#fdae61", lw=2.5, label="latest 60-min forecast")
    ax.fill_between(t, traj - 2 * sd, traj + 2 * sd, color="#fdae61", alpha=0.3, label="ensemble ±2 sd")

    alerts = df.loc[df.index >= begin]
    for flag, color, label in (("hypo_alert", "#7b3294", "hypo alert (30 min ahead)"),
                               ("hyper_alert", "#d7191c", "hyper alert (30 min ahead)")):
        hits = alerts.index[alerts[flag]]
        if len(hits):
            ax.scatter(hits, np.full(len(hits), 12 if flag == "hypo_alert" else 388), marker="|",
                       color=color, s=60, label=label)

    ax.set_ylim(0, 400)
    ax.set_ylabel("glucose (mg/dL)")
    ax.set_title("CGM glucose and forecasts")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=120)
        plt.close(fig)


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="directory written by glucose_prediction.train")
    p.add_argument("--input", nargs="+", required=True, help="XML file(s) of one patient to forecast")
    p.add_argument("--history", nargs="*", default=[],
                   help="earlier XML file(s) of the same patient, used only as context")
    p.add_argument("--output", help="CSV file for the forecasts")
    p.add_argument("--plot", help="save the plot to this file (omit to skip, '-' to show a window)")
    p.add_argument("--days", type=float, default=2.0, help="days of recorded data shown in the plot")
    args = p.parse_args(argv)

    model = GlucoseForecaster.load(args.model)
    df = forecast(model, args.input, args.history)
    w = df.attrs.pop("windows")

    if not np.all(np.isnan(w.y)):
        pred = df.filter(like="pred_").to_numpy()
        suite = {"persistence": lambda x: persistence_forecast(x, model.config.horizon),
                 "neural_net": lambda _: pred}
        print(format_report(evaluate_models(suite, w, model.config.horizon), REPORT_HORIZONS))

    latest = df.iloc[-1]
    print(f"\nLatest reading {df.index[-1]}: {latest['glucose']:.0f} mg/dL -> "
          f"30 min: {latest['pred_30min']:.0f}, 60 min: {latest['pred_60min']:.0f} mg/dL"
          + ("  [HYPO ALERT]" if latest["hypo_alert"] else "")
          + ("  [HYPER ALERT]" if latest["hyper_alert"] else ""))

    if args.output:
        df.to_csv(args.output)
        print(f"forecasts written to {args.output}")
    if args.plot:
        recorded = load_patient(list(args.history) + list(args.input))["glucose"]
        plot_forecast(df, recorded, None if args.plot == "-" else args.plot, days=args.days)
        if args.plot != "-":
            print(f"plot written to {args.plot}")
    return df


if __name__ == "__main__":
    main()
