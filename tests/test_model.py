import numpy as np
import pytest

from conftest import write_ohio_xml
from glucose_prediction.model import (GlucoseForecaster, ModelConfig, RidgeForecaster, Standardizer,
                                      linear_extrapolation_forecast, masked_huber, persistence_forecast)
from glucose_prediction.pipeline import build_test_windows, build_training_sets


@pytest.fixture
def datasets(tmp_path):
    train = write_ohio_xml(tmp_path / "train.xml", start="01-01-2022 00:02:00", days=4)
    test = write_ohio_xml(tmp_path / "test.xml", start="05-01-2022 00:02:00", days=1, seed=3)
    cfg = ModelConfig(history=12, horizon=6, units=8, dense_units=8, max_epochs=2, patience=1,
                      ensemble_size=2, batch_size=64)
    tr, va = build_training_sets([train], cfg, val_fraction=0.25)
    te = build_test_windows([test], cfg, context_paths=[train])["999"]
    return cfg, tr, va, te


def test_masked_huber_ignores_missing_targets():
    loss = masked_huber(1.0)
    y_true = np.array([[0.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    y_pred = np.array([[0.5, 100.0], [3.0, 3.0]], dtype=np.float32)
    out = np.asarray(loss(y_true, y_pred))
    assert out[0] == pytest.approx(0.125)
    assert out[1] == 0.0


def test_standardizer_roundtrip(datasets):
    _, tr, _, _ = datasets
    s = Standardizer().fit(tr)
    z = s.transform_y(tr.y, tr.last)
    np.testing.assert_allclose(s.inverse_y(z, tr.last), tr.y, rtol=1e-4, atol=1e-3)
    s2 = Standardizer.from_state(s.state())
    np.testing.assert_allclose(s2.transform_x(tr.X[:5]), s.transform_x(tr.X[:5]))


def test_baselines(datasets):
    cfg, tr, _, _ = datasets
    p = persistence_forecast(tr, cfg.horizon)
    assert p.shape == (len(tr), cfg.horizon)
    np.testing.assert_allclose(p[:, -1], tr.last)
    lin = linear_extrapolation_forecast(tr, cfg.horizon)
    assert np.isfinite(lin).all()
    ridge = RidgeForecaster(cfg).fit(tr)
    assert ridge.predict(tr).shape == (len(tr), cfg.horizon)


def test_test_windows_start_inside_test_file(datasets):
    _, _, _, te = datasets
    assert te.times.min() >= np.datetime64("2022-01-05T00:00")
    # With the training file as context the first test reading already has a full window.
    assert te.times.min() <= np.datetime64("2022-01-05T00:05")


def test_forecaster_train_save_load(datasets, tmp_path):
    cfg, tr, va, te = datasets
    model = GlucoseForecaster(cfg).fit(tr, va)
    pred = model.predict(te)
    assert pred.shape == (len(te), cfg.horizon)
    assert np.isfinite(pred).all()
    assert model.predict_members(te).shape == (2, len(te), cfg.horizon)

    model.save(tmp_path / "artifacts")
    loaded = GlucoseForecaster.load(tmp_path / "artifacts")
    np.testing.assert_allclose(loaded.predict(te), pred, rtol=1e-5, atol=1e-3)
