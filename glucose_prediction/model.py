"""Forecasting models: a recurrent neural network ensemble plus reference baselines."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURE_COLUMNS, Windows


@dataclass
class ModelConfig:
    history: int = 24  # input window, steps of 5 min (2 h)
    horizon: int = 12  # forecast steps (1 h); every step 5..60 min is predicted jointly
    architecture: str = "gru"  # gru | lstm | bilstm | cnn_lstm
    units: int = 32
    dense_units: int = 64
    dropout: float = 0.3
    l2: float = 1e-5
    learning_rate: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 15
    huber_delta: float = 1.0
    ensemble_size: int = 5
    seed: int = 42
    feature_columns: list = field(default_factory=lambda: list(FEATURE_COLUMNS))


class Standardizer:
    """Input and target scaling fitted on training windows only.

    The network predicts the change from the current glucose (``y - last``), which removes the
    dominant persistence component and makes the learning problem much better conditioned.
    """

    def fit(self, w: Windows) -> "Standardizer":
        flat = w.X.reshape(-1, w.X.shape[-1]).astype(np.float64)
        self.x_mean = flat.mean(axis=0)
        self.x_std = flat.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        delta = w.y - w.last[:, None]
        self.y_mean = np.nanmean(delta, axis=0)
        self.y_std = np.nanstd(delta, axis=0)
        self.y_std[~np.isfinite(self.y_std) | (self.y_std < 1e-8)] = 1.0
        self.y_mean[~np.isfinite(self.y_mean)] = 0.0
        return self

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def transform_y(self, y: np.ndarray, last: np.ndarray) -> np.ndarray:
        return ((y - last[:, None] - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, z: np.ndarray, last: np.ndarray) -> np.ndarray:
        return z * self.y_std + self.y_mean + last[:, None]

    def state(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "y_mean", "y_std")}

    @classmethod
    def from_state(cls, state: dict) -> "Standardizer":
        obj = cls()
        for k, v in state.items():
            setattr(obj, k, np.asarray(v, dtype=np.float64))
        return obj


# --------------------------------------------------------------------------- baselines

def persistence_forecast(w: Windows, horizon: int) -> np.ndarray:
    """Zero-order hold: future glucose equals the current reading."""
    return np.repeat(w.last[:, None], horizon, axis=1)


def linear_extrapolation_forecast(w: Windows, horizon: int, fit_steps: int = 6) -> np.ndarray:
    """Extrapolate a least-squares line through the last ``fit_steps`` glucose inputs (30 min)."""
    g = w.X[:, -fit_steps:, FEATURE_COLUMNS.index("glucose")].astype(np.float64)
    t = np.arange(fit_steps) - (fit_steps - 1)
    tc = t - t.mean()
    slope = ((g - g.mean(axis=1, keepdims=True)) * tc).sum(axis=1) / (tc**2).sum()
    steps = np.arange(1, horizon + 1)
    return w.last[:, None] + slope[:, None] * steps[None, :]


class RidgeForecaster:
    """Linear autoregressive model with exogenous inputs (ARX) on the same features."""

    def __init__(self, config: ModelConfig, alpha: float = 10.0):
        self.config = config
        self.alpha = alpha

    def fit(self, train: Windows, val: Windows | None = None) -> "RidgeForecaster":
        from sklearn.linear_model import Ridge

        self.scaler = Standardizer().fit(train)
        X = self.scaler.transform_x(train.X).reshape(len(train), -1)
        Z = self.scaler.transform_y(train.y, train.last)
        self.models = []
        for h in range(self.config.horizon):
            ok = ~np.isnan(Z[:, h])
            self.models.append(Ridge(alpha=self.alpha).fit(X[ok], Z[ok, h]))
        return self

    def predict(self, w: Windows) -> np.ndarray:
        X = self.scaler.transform_x(w.X).reshape(len(w), -1)
        Z = np.stack([m.predict(X) for m in self.models], axis=1)
        return self.scaler.inverse_y(Z, w.last)


# --------------------------------------------------------------------------- neural network

def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras  # noqa: WPS433 (deferred: TensorFlow import is slow)

    return keras


def masked_huber(delta: float):
    """Huber loss that ignores targets that are ``NaN`` (missing CGM readings)."""
    keras = _keras()
    ops = keras.ops

    def loss(y_true, y_pred):
        mask = ops.logical_not(ops.isnan(y_true))
        y_true = ops.where(mask, y_true, y_pred)
        err = ops.abs(y_true - y_pred)
        quad = ops.minimum(err, delta)
        per_elem = 0.5 * quad**2 + delta * (err - quad)
        maskf = ops.cast(mask, per_elem.dtype)
        return ops.sum(per_elem * maskf, axis=-1) / ops.maximum(ops.sum(maskf, axis=-1), 1.0)

    return loss


def build_network(config: ModelConfig, n_features: int):
    keras = _keras()
    layers = keras.layers
    reg = keras.regularizers.l2(config.l2) if config.l2 else None

    inputs = keras.Input(shape=(config.history, n_features))
    x = inputs
    arch = config.architecture
    if arch == "cnn_lstm":
        x = layers.Conv1D(config.units, 3, padding="causal", activation="relu", kernel_regularizer=reg)(x)
        x = layers.LSTM(config.units, kernel_regularizer=reg)(x)
    elif arch == "lstm":
        x = layers.LSTM(config.units, return_sequences=True, kernel_regularizer=reg)(x)
        x = layers.LSTM(config.units // 2, kernel_regularizer=reg)(x)
    elif arch == "gru":
        x = layers.GRU(config.units, return_sequences=True, kernel_regularizer=reg)(x)
        x = layers.GRU(config.units // 2, kernel_regularizer=reg)(x)
    elif arch == "bilstm":
        # Bidirectional recurrence only runs over the *past* input window, so it is causal.
        x = layers.Bidirectional(layers.LSTM(config.units, return_sequences=True, kernel_regularizer=reg))(x)
        x = layers.Bidirectional(layers.LSTM(config.units // 2, kernel_regularizer=reg))(x)
    else:
        raise ValueError(f"unknown architecture {arch!r}")

    # Skip connection: the most recent input step carries most of the short-term information.
    last_step = layers.Flatten()(layers.Cropping1D((config.history - 1, 0))(inputs))
    x = layers.Concatenate()([x, last_step])
    x = layers.Dropout(config.dropout)(x)
    x = layers.Dense(config.dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(config.dropout)(x)
    outputs = layers.Dense(config.horizon)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate, clipnorm=1.0),
                  loss=masked_huber(config.huber_delta))
    return model


class GlucoseForecaster:
    """Ensemble of recurrent networks predicting glucose 5..60 min ahead."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.networks = []
        self.histories = []
        self.scaler: Standardizer | None = None

    def fit(self, train: Windows, val: Windows, verbose: int = 0) -> "GlucoseForecaster":
        keras = _keras()
        cfg = self.config
        self.scaler = Standardizer().fit(train)
        Xtr, ytr = self.scaler.transform_x(train.X), self.scaler.transform_y(train.y, train.last)
        Xva, yva = self.scaler.transform_x(val.X), self.scaler.transform_y(val.y, val.last)

        self.networks, self.histories = [], []
        for member in range(cfg.ensemble_size):
            keras.utils.set_random_seed(cfg.seed + member)
            net = build_network(cfg, Xtr.shape[-1])
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                  patience=max(cfg.patience // 3, 2), min_lr=1e-5),
            ]
            hist = net.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=cfg.max_epochs,
                           batch_size=cfg.batch_size, shuffle=True, callbacks=callbacks, verbose=verbose)
            self.networks.append(net)
            self.histories.append(hist.history)
        return self

    def predict_members(self, w: Windows) -> np.ndarray:
        """Predictions of each ensemble member, shape ``(members, n, horizon)`` in mg/dL."""
        X = self.scaler.transform_x(w.X)
        return np.stack([self.scaler.inverse_y(net.predict(X, batch_size=1024, verbose=0), w.last)
                         for net in self.networks])

    def predict(self, w: Windows) -> np.ndarray:
        return self.predict_members(w).mean(axis=0)

    # ---------------------------------------------------------------- persistence
    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        (d / "scaler.json").write_text(json.dumps(self.scaler.state()))
        (d / "history.json").write_text(json.dumps(self.histories, default=float))
        for i, net in enumerate(self.networks):
            net.save(d / f"network_{i}.keras")

    @classmethod
    def load(cls, directory) -> "GlucoseForecaster":
        keras = _keras()
        d = Path(directory)
        obj = cls(ModelConfig(**json.loads((d / "config.json").read_text())))
        if obj.config.feature_columns != FEATURE_COLUMNS:
            raise ValueError("model was trained with a different feature set than this code version")
        obj.scaler = Standardizer.from_state(json.loads((d / "scaler.json").read_text()))
        paths = sorted(d.glob("network_*.keras"), key=lambda p: int(p.stem.split("_")[1]))
        obj.networks = [keras.saving.load_model(p, compile=False) for p in paths]
        if not obj.networks:
            raise FileNotFoundError(f"no network_*.keras files in {d}")
        return obj
