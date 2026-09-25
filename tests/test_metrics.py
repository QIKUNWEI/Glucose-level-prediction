import numpy as np
import pytest

from glucose_prediction.metrics import clarke_zones, regression_metrics


@pytest.mark.parametrize("ref,pred,zone", [
    (100, 100, "A"),
    (100, 118, "A"),
    (60, 50, "A"),  # both hypoglycaemic
    (100, 130, "B"),
    (100, 215, "C"),
    (50, 100, "D"),  # missed hypoglycaemia
    (300, 150, "D"),  # missed hyperglycaemia
    (250, 60, "E"),
    (60, 200, "E"),
])
def test_clarke_zones(ref, pred, zone):
    assert clarke_zones(np.array([ref]), np.array([pred]))[0] == zone


def test_regression_metrics_perfect_and_nan():
    ref = np.array([60.0, 100.0, 200.0, np.nan])
    m = regression_metrics(ref, ref.copy())
    assert m["n"] == 3
    assert m["rmse"] == 0 and m["mae"] == 0 and m["clarke_A_pct"] == 100
    assert m["hypo_alert"]["sensitivity"] == 1.0


def test_regression_metrics_values():
    m = regression_metrics([100, 200], [110, 180])
    assert m["mae"] == pytest.approx(15)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["mard_pct"] == pytest.approx(10)
    assert m["bias"] == pytest.approx(-5)
