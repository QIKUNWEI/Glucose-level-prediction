"""Parsing of OhioT1DM XML files and resampling onto a regular 5-minute grid.

The OhioT1DM XML layout stores every data stream in its own top-level element, e.g.::

    <patient id="559" weight="99" insulin_type="Novalog">
      <glucose_level><event ts="07-12-2021 01:17:00" value="101"/> ...</glucose_level>
      <basal><event ts="..." value="0.65"/> ...</basal>
      <temp_basal><event ts_begin="..." ts_end="..." value="0.0"/> ...</temp_basal>
      <bolus><event ts_begin="..." ts_end="..." type="normal" dose="1.6" .../> ...</bolus>
      <meal><event ts="..." type="Snack" carbs="18"/> ...</meal>
      <exercise><event ts="..." intensity="3" duration="150" .../> ...</exercise>
      ...
    </patient>

Meals, insulin and exercise are *not* children of the glucose events, so they have to be
read from their own sections and aligned to the CGM timeline by timestamp.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TS_FORMAT = "%d-%m-%Y %H:%M:%S"
FREQ = "5min"
STEP_MINUTES = 5

# Physiologically plausible CGM range (Medtronic Enlite sensors report 40-400 mg/dL).
GLUCOSE_MIN = 40.0
GLUCOSE_MAX = 400.0

# Maximum look-back for insulin/meal events that precede the first CGM reading.
LEAD_IN = pd.Timedelta(hours=6)

GRID_COLUMNS = ["glucose", "basal_insulin", "bolus_insulin", "carbs", "exercise"]


@dataclass
class PatientRecord:
    """Raw event streams of one OhioT1DM XML file."""

    patient_id: str
    glucose: pd.Series  # mg/dL indexed by timestamp
    basal: pd.DataFrame = field(default_factory=pd.DataFrame)  # ts, value (U/h)
    temp_basal: pd.DataFrame = field(default_factory=pd.DataFrame)  # ts_begin, ts_end, value (U/h)
    bolus: pd.DataFrame = field(default_factory=pd.DataFrame)  # ts_begin, ts_end, dose (U)
    meal: pd.DataFrame = field(default_factory=pd.DataFrame)  # ts, carbs (g)
    exercise: pd.DataFrame = field(default_factory=pd.DataFrame)  # ts, intensity, duration (min)


def _parse_ts(values) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(values, dtype="object"), format=TS_FORMAT, errors="coerce")


def _section(root: ET.Element, tag: str, attrs: dict[str, str]) -> pd.DataFrame:
    """Collect the events of one section into a DataFrame.

    ``attrs`` maps attribute names to their kind: ``"ts"`` (timestamp), ``"float"`` or ``"str"``.
    Rows with an unparsable timestamp or numeric attribute are dropped.
    """
    node = root.find(tag)
    rows = [] if node is None else [ev.attrib for ev in node.findall("event")]
    df = pd.DataFrame(rows, columns=list(attrs))
    for name, kind in attrs.items():
        if kind == "ts":
            df[name] = _parse_ts(df[name])
        elif kind == "float":
            df[name] = pd.to_numeric(df[name], errors="coerce")
    required = [n for n, k in attrs.items() if k != "str"]
    return df.dropna(subset=required).reset_index(drop=True)


def parse_ohio_xml(path) -> PatientRecord:
    """Parse an OhioT1DM XML file into its raw event streams."""
    root = ET.parse(path).getroot()

    cgm = _section(root, "glucose_level", {"ts": "ts", "value": "float"})
    glucose = cgm.set_index("ts")["value"].sort_index()
    glucose.index.name = "timestamp"

    return PatientRecord(
        patient_id=root.get("id", "unknown"),
        glucose=glucose,
        basal=_section(root, "basal", {"ts": "ts", "value": "float"}).sort_values("ts"),
        temp_basal=_section(root, "temp_basal", {"ts_begin": "ts", "ts_end": "ts", "value": "float"}),
        bolus=_section(root, "bolus", {"ts_begin": "ts", "ts_end": "ts", "dose": "float", "type": "str"}),
        meal=_section(root, "meal", {"ts": "ts", "carbs": "float", "type": "str"}),
        exercise=_section(root, "exercise", {"ts": "ts", "intensity": "float", "duration": "float"}),
    )


def _spread(minute_index: pd.DatetimeIndex, begins, ends, totals) -> np.ndarray:
    """Distribute ``totals`` uniformly over [begin, end) on a 1-minute index.

    Instantaneous events (end <= begin) are placed entirely in the minute of ``begin``.
    """
    out = np.zeros(len(minute_index))
    start = minute_index[0]
    n = len(out)
    for b, e, total in zip(begins, ends, totals):
        i0 = int((b.floor("min") - start) / pd.Timedelta(minutes=1))
        i1 = int((e.floor("min") - start) / pd.Timedelta(minutes=1)) if pd.notna(e) else i0
        i1 = max(i1, i0 + 1)
        lo, hi = max(i0, 0), min(i1, n)
        if hi <= lo:
            continue
        out[lo:hi] += total / (i1 - i0)
    return out


def build_timeseries(record: PatientRecord) -> pd.DataFrame:
    """Resample a :class:`PatientRecord` onto a regular 5-minute grid.

    Columns
    -------
    glucose        CGM reading in mg/dL; ``NaN`` where the sensor produced no value.
    basal_insulin  basal insulin delivered in the 5-min bin (U), temp basals applied.
    bolus_insulin  bolus insulin delivered in the bin (U); square/dual boluses are spread
                   over their delivery interval.
    carbs          carbohydrates eaten in the bin (g).
    exercise       mean self-reported exercise intensity (0-10) during the bin.
    """
    glucose = record.glucose.clip(GLUCOSE_MIN, GLUCOSE_MAX)
    if glucose.empty:
        raise ValueError(f"Patient {record.patient_id}: no glucose readings found")

    # CGM readings arrive every ~5 min but are not phase-aligned to the clock and the phase
    # can shift after a sensor change; snap each reading to the nearest 5-min grid point.
    snapped = glucose.groupby(glucose.index.round(FREQ)).mean()
    # Start the grid a little before the first reading when insulin or meals were logged
    # earlier, so that they still count towards insulin/carbs on board.
    event_starts = [s.min() for s in (record.bolus.get("ts_begin"), record.meal.get("ts"))
                    if s is not None and len(s)]
    start = snapped.index.min()
    if event_starts:
        start = max(min(start, min(event_starts).floor(FREQ)), start - LEAD_IN)
    grid = pd.date_range(start, snapped.index.max(), freq=FREQ, name="timestamp")
    df = pd.DataFrame(index=grid)
    df["glucose"] = snapped.reindex(grid)

    minutes = pd.date_range(grid[0], grid[-1] + pd.Timedelta(minutes=STEP_MINUTES - 1), freq="min")
    bins = minutes.floor(FREQ)

    # Basal: step function of the programmed rate (U/h), overridden by temporary basals.
    rate = pd.Series(np.nan, index=minutes)
    if not record.basal.empty:
        programmed = record.basal.groupby("ts")["value"].last()
        combined = programmed.reindex(programmed.index.union(minutes)).ffill()
        rate = combined.reindex(minutes)
        # Carry the programmed rate in force before the grid starts.
        before = programmed[programmed.index <= minutes[0]]
        if len(before):
            rate = rate.fillna(before.iloc[-1])
    rate = rate.fillna(0.0).to_numpy(dtype=float, copy=True)
    for tb in record.temp_basal.itertuples():
        mask = (minutes >= tb.ts_begin) & (minutes < tb.ts_end)
        rate[mask] = tb.value
    df["basal_insulin"] = pd.Series(rate / 60.0, index=minutes).groupby(bins).sum().reindex(grid).to_numpy()

    bolus = _spread(minutes, record.bolus.get("ts_begin", []), record.bolus.get("ts_end", []),
                    record.bolus.get("dose", []))
    df["bolus_insulin"] = pd.Series(bolus, index=minutes).groupby(bins).sum().reindex(grid).to_numpy()

    carbs = _spread(minutes, record.meal.get("ts", []), record.meal.get("ts", []), record.meal.get("carbs", []))
    df["carbs"] = pd.Series(carbs, index=minutes).groupby(bins).sum().reindex(grid).to_numpy()

    ex = record.exercise
    ends = [b + pd.Timedelta(minutes=max(d, 1.0)) for b, d in zip(ex.get("ts", []), ex.get("duration", []))]
    # Spreading intensity*duration over the duration yields the intensity for each covered minute.
    ex_total = [i * max(d, 1.0) for i, d in zip(ex.get("intensity", []), ex.get("duration", []))]
    exercise = _spread(minutes, ex.get("ts", []), ends, ex_total)
    df["exercise"] = pd.Series(exercise, index=minutes).groupby(bins).mean().reindex(grid).to_numpy()

    df.attrs["patient_id"] = record.patient_id
    return df[GRID_COLUMNS]


def load_patient(paths) -> pd.DataFrame:
    """Load one or more consecutive XML files of the *same* patient into a single grid.

    Passing ``[training_xml, testing_xml]`` gives a continuous timeline in which the tail of
    the training file provides the history (insulin on board, input window) for the first
    test predictions. Use :func:`split_at` or a timestamp filter to evaluate only test points.
    """
    if isinstance(paths, (str, bytes)) or not isinstance(paths, (list, tuple)):
        paths = [paths]
    records = [parse_ohio_xml(p) for p in paths]
    ids = {r.patient_id for r in records}
    if len(ids) > 1:
        raise ValueError(f"load_patient expects files of a single patient, got ids {sorted(ids)}")
    if len(records) == 1:
        return build_timeseries(records[0])

    merged = PatientRecord(
        patient_id=records[0].patient_id,
        glucose=pd.concat([r.glucose for r in records]).sort_index(),
        basal=pd.concat([r.basal for r in records]).sort_values("ts"),
        temp_basal=pd.concat([r.temp_basal for r in records]),
        bolus=pd.concat([r.bolus for r in records]),
        meal=pd.concat([r.meal for r in records]),
        exercise=pd.concat([r.exercise for r in records]),
    )
    merged.glucose = merged.glucose[~merged.glucose.index.duplicated(keep="last")]
    return build_timeseries(merged)
