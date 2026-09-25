"""Train the glucose forecaster and benchmark it against baselines.

Example (OhioT1DM, patient 559)::

    python -m glucose_prediction.train \
        --train data/559-ws-training.xml \
        --test data/559-ws-testing.xml \
        --out artifacts/559
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

from .features import Windows, add_features, make_windows
from .model import GlucoseForecaster, ModelConfig, RidgeForecaster
from .pipeline import (REPORT_HORIZONS, build_test_windows, build_training_sets, evaluate_models,
                       format_report, standard_model_suite)
from .data import load_patient


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", required=True,
                   help="training XML files (any number of patients); each patient is split chronologically")
    p.add_argument("--extra-train", nargs="*", default=[],
                   help="XML files used *only* as additional training data (e.g. other patients)")
    p.add_argument("--test", nargs="*", default=[],
                   help="held-out XML files; the matching training file supplies history")
    p.add_argument("--out", default="artifacts", help="output directory for the model and reports")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="final fraction of each training timeline used for early stopping")
    defaults = ModelConfig()
    for f in fields(ModelConfig):
        if f.name == "feature_columns":
            continue
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(getattr(defaults, f.name)),
                       default=getattr(defaults, f.name))
    p.add_argument("--verbose", type=int, default=0, help="Keras verbosity")
    p.add_argument("--plot", action="store_true", help="save training curves to <out>/training_curves.png")
    return p.parse_args(argv)


def plot_training_curves(histories, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, h in enumerate(histories):
        ax.plot(h["loss"], color=f"C{i}", alpha=0.8, label=f"member {i} train")
        ax.plot(h["val_loss"], color=f"C{i}", linestyle="--", label=f"member {i} validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked Huber loss (standardised)")
    ax.set_title("Training curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> dict:
    args = parse_args(argv)
    config = ModelConfig(**{f.name: getattr(args, f.name) for f in fields(ModelConfig) if f.name != "feature_columns"})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, val = build_training_sets(args.train, config, args.val_fraction)
    if args.extra_train:
        extra = [make_windows(add_features(load_patient(p)), config.history, config.horizon)
                 for p in args.extra_train]
        train = Windows.concat([train] + extra)
    print(f"training windows: {len(train)}, validation windows: {len(val)}")

    ridge = RidgeForecaster(config).fit(train)
    net = GlucoseForecaster(config).fit(train, val, verbose=args.verbose)
    net.save(out)
    if args.plot:
        plot_training_curves(net.histories, out / "training_curves.png")

    suite = standard_model_suite(net, ridge, config.horizon)
    results = {"config": asdict(config), "validation": evaluate_models(suite, val, config.horizon)}
    print("\nValidation (chronological hold-out of the training files)")
    print(format_report(results["validation"]))

    if args.test:
        results["test"] = {}
        for pid, w in build_test_windows(args.test, config, context_paths=args.train).items():
            results["test"][pid] = evaluate_models(suite, w, config.horizon)
            print(f"\nTest, patient {pid}")
            print(format_report(results["test"][pid]))

    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nModel and metrics written to {out}/ (horizons reported: "
          f"{', '.join(str(h * 5) + ' min' for h in REPORT_HORIZONS)})")
    return results


if __name__ == "__main__":
    main()
