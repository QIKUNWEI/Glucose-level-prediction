import numpy as np
import pandas as pd
import pytest

from conftest import write_ohio_xml
from glucose_prediction.data import build_timeseries, load_patient, parse_ohio_xml


def test_parser_reads_all_event_sections(xml_file):
    rec = parse_ohio_xml(xml_file)
    assert rec.patient_id == "999"
    assert len(rec.glucose) == 576
    assert len(rec.meal) == 6
    assert len(rec.bolus) == 7
    assert len(rec.temp_basal) == 1
    assert len(rec.exercise) == 1
    assert rec.glucose.index.is_monotonic_increasing


def test_timeseries_conserves_insulin_and_carbs(xml_file):
    rec = parse_ohio_xml(xml_file)
    grid = build_timeseries(rec)
    assert grid.index.freq == pd.Timedelta("5min")
    assert grid["bolus_insulin"].sum() == pytest.approx(rec.bolus["dose"].sum())
    assert grid["carbs"].sum() == pytest.approx(rec.meal["carbs"].sum())


def test_square_bolus_is_spread_over_its_duration(xml_file):
    grid = build_timeseries(parse_ohio_xml(xml_file))
    window = grid.loc["2022-01-01 22:00":"2022-01-01 22:55", "bolus_insulin"]
    assert len(window) == 12
    np.testing.assert_allclose(window.to_numpy(), 0.1)


def test_basal_schedule_and_temp_basal(xml_file):
    grid = build_timeseries(parse_ohio_xml(xml_file))
    day1 = grid.loc["2022-01-01", "basal_insulin"]
    # 0.6 U/h for 00-02h and 03-06h, temp basal 0 U/h for 02-03h, 1.2 U/h for 06-24h.
    assert day1.sum() == pytest.approx(0.6 * 5 + 1.2 * 18)
    assert grid.loc["2022-01-01 02:30", "basal_insulin"] == 0.0
    assert grid.loc["2022-01-01 07:00", "basal_insulin"] == pytest.approx(1.2 / 12)


def test_exercise_intensity_covers_duration(xml_file):
    grid = build_timeseries(parse_ohio_xml(xml_file))
    ex = grid["exercise"]
    assert (ex > 0).sum() == 6
    assert ex.max() == pytest.approx(5.0)


def test_cgm_gaps_become_nan(tmp_path):
    path = write_ohio_xml(tmp_path / "gap.xml", gap=(100, 10))
    grid = build_timeseries(parse_ohio_xml(path))
    assert grid["glucose"].isna().sum() == 10


def test_load_patient_joins_consecutive_files(tmp_path):
    a = write_ohio_xml(tmp_path / "a.xml", start="01-01-2022 00:02:00", days=1)
    b = write_ohio_xml(tmp_path / "b.xml", start="02-01-2022 00:02:00", days=1, seed=1)
    grid = load_patient([b, a])
    assert grid.index.is_monotonic_increasing
    assert grid["glucose"].notna().sum() == 576
    assert grid["carbs"].sum() == pytest.approx(2 * 170)


def test_load_patient_rejects_mixed_patients(tmp_path):
    a = write_ohio_xml(tmp_path / "a.xml", patient_id="1")
    b = write_ohio_xml(tmp_path / "b.xml", patient_id="2")
    with pytest.raises(ValueError):
        load_patient([a, b])
