import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TS = "%d-%m-%Y %H:%M:%S"


def write_ohio_xml(path, start="01-01-2022 00:02:00", days=2.0, patient_id="999", seed=0, gap=None):
    """Write a small synthetic file in the OhioT1DM XML layout.

    ``gap`` is an optional ``(start_step, n_steps)`` range of CGM readings to leave out.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(pd.to_datetime(start, format=TS))
    n = int(days * 288)
    minutes = np.arange(n) * 5
    glucose = 140 + 50 * np.sin(2 * np.pi * minutes / (24 * 60)) + rng.normal(0, 5, n)
    glucose = np.clip(glucose, 40, 400)

    def ts(t):
        return t.strftime(TS)

    cgm = []
    for i in range(n):
        if gap and gap[0] <= i < gap[0] + gap[1]:
            continue
        cgm.append(f'<event ts="{ts(t0 + pd.Timedelta(minutes=int(minutes[i])))}" value="{glucose[i]:.0f}"/>')
    day0 = t0.normalize()
    meals, boluses, basals = [], [], []
    for d in range(int(np.ceil(days))):
        for hour, carbs in ((8, 40), (13, 60), (19, 70)):
            t = day0 + pd.Timedelta(days=d, hours=hour)
            meals.append(f'<event ts="{ts(t)}" type="Meal" carbs="{carbs}"/>')
            boluses.append(f'<event ts_begin="{ts(t)}" ts_end="{ts(t)}" type="normal" dose="{carbs / 10}" '
                           f'bwz_carb_input="{carbs}"/>')
        basals.append(f'<event ts="{ts(day0 + pd.Timedelta(days=d))}" value="0.6"/>')
        basals.append(f'<event ts="{ts(day0 + pd.Timedelta(days=d, hours=6))}" value="1.2"/>')
    temp = (f'<event ts_begin="{ts(day0 + pd.Timedelta(hours=2))}" '
            f'ts_end="{ts(day0 + pd.Timedelta(hours=3))}" value="0.0"/>')
    square = (f'<event ts_begin="{ts(day0 + pd.Timedelta(hours=22))}" '
              f'ts_end="{ts(day0 + pd.Timedelta(hours=23))}" type="square" dose="1.2"/>')
    exercise = f'<event ts="{ts(day0 + pd.Timedelta(hours=17))}" intensity="5" type=" " duration="30" competitive=""/>'

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<patient id="{patient_id}" weight="99" insulin_type="Novalog">
  <glucose_level>{''.join(cgm)}</glucose_level>
  <finger_stick></finger_stick>
  <basal>{''.join(basals)}</basal>
  <temp_basal>{temp}</temp_basal>
  <bolus>{''.join(boluses)}{square}</bolus>
  <meal>{''.join(meals)}</meal>
  <exercise>{exercise}</exercise>
</patient>
"""
    Path(path).write_text(xml)
    return path


@pytest.fixture
def xml_file(tmp_path):
    return write_ohio_xml(tmp_path / "999-ws-training.xml")
