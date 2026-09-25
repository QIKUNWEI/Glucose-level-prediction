import numpy as np
import pandas as pd
import pytest

from conftest import write_ohio_xml
from glucose_prediction.data import build_timeseries, parse_ohio_xml
from glucose_prediction.features import (FEATURE_COLUMNS, add_features, causal_interpolate,
                                         exponential_action_curves, make_windows)
from glucose_prediction.pipeline import chronological_split


def test_action_curve_properties():
    activity, remaining = exponential_action_curves(360, 75)
    assert remaining[0] == pytest.approx(1.0)
    assert remaining[-1] == 0.0
    assert np.all(np.diff(remaining) <= 1e-12)
    assert activity.sum() == pytest.approx(1.0)
    assert np.argmax(activity) * 5 == pytest.approx(75, abs=5)


def test_causal_interpolate_limits():
    s = pd.Series([100, np.nan, np.nan, 130] + [np.nan] * 8 + [150, np.nan])
    out = causal_interpolate(s, limit=6)
    assert out.iloc[1:3].tolist() == [110, 120]
    assert out.iloc[4:12].isna().all()  # gap longer than the limit stays missing
    assert np.isnan(out.iloc[-1])  # never extrapolate past the last reading


def test_windows_do_not_depend_on_the_future(xml_file):
    grid = build_timeseries(parse_ohio_xml(xml_file))
    feats = add_features(grid)
    w = make_windows(feats, history=24, horizon=12, require_target=False)

    cut = grid.index[300]
    altered = grid.copy()
    after = altered.index > cut
    altered.loc[after, "glucose"] = 400.0
    altered.loc[after, "carbs"] += 100.0
    altered.loc[after, "bolus_insulin"] += 5.0
    w2 = make_windows(add_features(altered), history=24, horizon=12, require_target=False)

    past = np.asarray(w.times <= cut)
    past2 = np.asarray(w2.times <= cut)
    assert past.sum() == past2.sum() > 0
    np.testing.assert_array_equal(w.X[past], w2.X[past2])


def test_window_shapes_and_targets(xml_file):
    feats = add_features(build_timeseries(parse_ohio_xml(xml_file)))
    w = make_windows(feats, history=24, horizon=12)
    assert w.X.shape[1:] == (24, len(FEATURE_COLUMNS))
    g = FEATURE_COLUMNS.index("glucose")
    np.testing.assert_allclose(w.X[:, -1, g], w.last)
    # y[i, h-1] is the reading h steps after the anchor.
    i = 10
    pos = feats.index.get_loc(w.times[i])
    np.testing.assert_allclose(w.y[i], feats["target"].iloc[pos + 1: pos + 13].to_numpy())


def test_windows_skip_long_gaps_and_imputed_anchors(tmp_path):
    path = write_ohio_xml(tmp_path / "gap.xml", gap=(200, 12))
    feats = add_features(build_timeseries(parse_ohio_xml(path)))
    w = make_windows(feats, history=24, horizon=12)
    gap_times = feats.index[200:212]
    assert not w.times.isin(gap_times).any()
    # Anchors right after a 60-min gap need 24 contiguous inputs, so none within 23 steps.
    after = feats.index[212:212 + 23]
    assert not w.times.isin(after).any()

    short = write_ohio_xml(tmp_path / "short.xml", gap=(200, 3))
    feats = add_features(build_timeseries(parse_ohio_xml(short)))
    w = make_windows(feats, history=24, horizon=12)
    assert not w.times.isin(feats.index[200:203]).any()  # interpolated points are never anchors
    assert feats.index[203:226].isin(w.times).all()  # but they may be inputs
    assert np.isnan(w.y).any()  # missing targets are kept as NaN, not interpolated


def test_chronological_split_has_embargo(xml_file):
    feats = add_features(build_timeseries(parse_ohio_xml(xml_file)))
    w = make_windows(feats, history=24, horizon=12)
    tr, va = chronological_split(w, 0.25, horizon=12)
    # The last training target (anchor + 60 min) precedes every validation target.
    assert tr.times.max() + pd.Timedelta(minutes=60) <= va.times.min()
    assert len(va) == pytest.approx(0.25 * len(w), abs=2)
